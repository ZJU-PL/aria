# vim: set sw=4 ts=4 softtabstop=4 expandtab:
import io
import logging
import z3
import aria.smt.fp.smt2coral.util as Util

_logger = logging.getLogger(__name__)


class CoralPrinterException(Exception):
    def __init__(self, msg):
        self.msg = msg


class CoralPrinterUnsupportedOperation(CoralPrinterException):
    def __init__(self, op):
        super().__init__(f"Unsupported operation:{op}")


class CoralPrinterUnsupportedRoundingMode(CoralPrinterException):
    def __init__(self, rm):
        super().__init__(f"Unsupported rounding mode:{rm}")


class CoralPrinterUnsupportedSort(CoralPrinterException):
    def __init__(self, s):
        super().__init__(f"Unsupported sort:{s}")


class CoralPrinter(Util.Z3ExprDispatcher):
    def __init__(self):
        super().__init__()
        self.sio = io.StringIO("")
        self.symbol_table = {}
        self._unsound_translation_occured = False

    def translation_was_sound(self):
        return not self._unsound_translation_occured

    def _unsound_translation(self, op):
        self._unsound_translation_occured = True
        _logger.warning("Unsound translation for %s operation", op)

    def print_constraints(self, constraints):
        assert isinstance(constraints, list)
        sio = io.StringIO("")
        for index, constraint in enumerate(constraints):
            self.print_constraint(constraint)
            if index < (len(constraints) - 1):
                self.sio.write(";")
        final_str = self.sio.getvalue()
        return final_str

    def reset(self):
        self.sio.close()
        self.sio = io.StringIO("")
        self.symbol_table = {}
        self._unsound_translation_occured = False

    def print_constraint(self, constraint):
        self.visit(constraint)

    # Visitors
    def visit_true(self, e):
        self.sio.write("BCONST(TRUE)")

    def visit_false(self, e):
        self.sio.write("BCONST(FALSE)")

    def _expr_is_supported_fp_sort(self, e):
        return self._is_supported_fp_sort(e.sort())

    def _is_supported_fp_sort(self, sort):
        if sort.kind() != z3.Z3_FLOATING_POINT_SORT:
            return False
        return self._is_float32_sort(sort) or self._is_float64_sort(sort)

    def _is_float32_sort(self, sort):
        assert sort.kind() == z3.Z3_FLOATING_POINT_SORT
        return sort.ebits() == 8 and sort.sbits() == 24

    def _is_float64_sort(self, sort):
        assert sort.kind() == z3.Z3_FLOATING_POINT_SORT
        return sort.ebits() == 11 and sort.sbits() == 53

    def _is_supported_bv_sort(self, sort):
        if sort.kind() != z3.Z3_BV_SORT:
            return False
        # TODO
        return False

    def _expr_is_supported_bv_sort(self, e):
        return self._is_supported_bv_sort(e.sort())

    def _check_fp_sort(self, e):
        if not self._expr_is_supported_fp_sort(e):
            raise CoralPrinterUnsupportedSort(e.sort())

    def _check_bv_sort(self, e):
        if not self._expr_is_supported_bv_sort(e):
            raise CoralPrinterUnsupportedSort(e.sort())

    def escape_variable_name(self, name):
        sym = None
        try:
            sym = self.symbol_table[name]
        except KeyError:
            sym = "ID_{}".format(len(self.symbol_table))
            self.symbol_table[name] = sym
        assert sym is not None
        return sym

    def visit_variable(self, e):
        sort = e.sort()
        decl = e.decl()
        name = decl.name()
        escaped_name = self.escape_variable_name(name)
        if sort.kind() == z3.Z3_BOOL_SORT:
            # FIXME: This doesn't really work. Coral doesn't document
            # this in its grammar but its parser does accept it. However
            # the rest of Coral's code seems to crash on this.
            _logger.warning("Emitting BVAR, coral will likely crash on this")
            self.sio.write("BVAR({})".format(escaped_name))
        elif sort.kind() == z3.Z3_BV_SORT:
            self._check_bv_sort(e)
            raise NotImplementedError("BitVector variable")
        elif sort.kind() == z3.Z3_FLOATING_POINT_SORT:
            self._check_fp_sort(e)
            if self._is_float32_sort(sort):
                self.sio.write("FVAR({})".format(escaped_name))
            elif self._is_float64_sort(sort):
                self.sio.write("DVAR({})".format(escaped_name))
            else:
                raise CoralPrinterUnsupportedSort(sort)
        else:
            raise CoralPrinterUnsupportedSort(sort)

    def _visit_binary_op(self, e, name):
        assert e.num_args() == 2
        self.sio.write(name + "(")
        self.visit(e.arg(0))
        self.sio.write(",")
        self.visit(e.arg(1))
        self.sio.write(")")

    def _visit_unary_op(self, e, name):
        assert e.num_args() == 1
        self.sio.write(name + "(")
        self.visit(e.arg(0))
        self.sio.write(")")

    def _visit_bool_nary_op(self, e, name):
        """
        Visit boolean operator that has an n-ary form
        """
        assert e.num_args() > 0
        if e.num_args() == 1:
            # Apparently Z3 allows this, just ignore
            # the operator and walk through it
            self.visit(e.arg(0))
        elif e.num_args() == 2:
            self._visit_binary_op(e, name)
        else:
            # Repeatedly apply the binary operator
            # to get the same semantics
            num_args = e.num_args()
            for index in range(0, num_args - 1):
                self.sio.write(name + "(")
                self.visit(e.arg(index))
                self.sio.write(",")
                if index == num_args - 2:
                    self.visit(e.arg(index + 1))
            # Emit closing brackets
            self.sio.write(")" * (num_args - 1))

    def visit_and(self, e):
        self._visit_bool_nary_op(e, "BAND")

    def visit_or(self, e):
        self._visit_bool_nary_op(e, "BOR")

    def visit_xor(self, e):
        self._visit_binary_op(e, "BXOR")

    def visit_not(self, e):
        self._visit_unary_op(e, "BNOT")

    def visit_implies(self, e):
        temp = z3.Or(z3.Not(e.arg(0)), e.arg(1))
        self.visit(temp)

    def visit_eq(self, e):
        sort = e.arg(0).sort()
        if sort.kind() == z3.Z3_BOOL_SORT:
            self.sio.write("BNOT(BXOR(")
            self.visit(e.arg(0))
            self.sio.write(",")
            self.visit(e.arg(1))
            self.sio.write("))")
        elif sort.kind() == z3.Z3_BV_SORT:
            self._check_bv_sort(e.arg(0))
            self._check_bv_sort(e.arg(1))
            raise NotImplementedError("BitVector equal")
        elif sort.kind() == z3.Z3_FLOATING_POINT_SORT:
            self._check_fp_sort(e.arg(0))
            # Either FEQ or both args are NaN
            # FIXME: This isn't quite right because +zero is != to -zero
            # but we have no way in Coral's constraint language of distinguishing
            # between them
            self._unsound_translation("=")
            new_expr = z3.Or(
                z3.And(z3.fpIsNaN(e.arg(0)), z3.fpIsNaN(e.arg(1))),
                z3.fpEQ(e.arg(0), e.arg(1)),
            )
            self.visit(new_expr)
        else:
            raise CoralPrinterUnsupportedSort(sort)

    def visit_ite(self, e):
        assert e.num_args() == 3
        # FIXME: Coral doesn't seem to support this fundemental operation :(
        raise CoralPrinterUnsupportedOperation("ite")

    def visit_distinct(self, e):
        assert e.num_args() >= 2
        # Construct a new expr that does an O(n^2) comparisons
        exprs_to_and = []
        for index_arg0 in range(0, e.num_args()):
            for index_arg1 in range(0, e.num_args()):
                if index_arg0 >= index_arg1:
                    # Skip unnecessary comparisons
                    continue
                new_expr = z3.Not(e.arg(index_arg0) == e.arg(index_arg1))
                exprs_to_and.append(new_expr)
        # Special case a single comparision
        if len(exprs_to_and) == 1:
            self.visit(exprs_to_and[0])
            return
        # N > 1 comparisons
        final_expr = z3.BoolVal(True)
        while len(exprs_to_and) > 0:
            e = exprs_to_and.pop()
            final_expr = z3.And(final_expr, e)
        self.visit(final_expr)

    def visit_float_plus_zero(self, e):
        assert e.num_args() == 0
        self._check_fp_sort(e)
        if self._is_float32_sort(e.sort()):
            self.sio.write("FCONST(0.0)")
        elif self._is_float64_sort(e.sort()):
            self.sio.write("DCONST(0.0)")
        else:
            raise CoralPrinterException("Unhandled +zero")

    def visit_float_minus_zero(self, e):
        assert e.num_args() == 0
        self._check_fp_sort(e)
        if self._is_float32_sort(e.sort()):
            self.sio.write("FCONST(-0.0)")
        elif self._is_float64_sort(e.sort()):
            self.sio.write("DCONST(-0.0)")
        else:
            raise CoralPrinterException("Unhandled -zero")

    def visit_float_plus_inf(self, e):
        assert e.num_args() == 0
        self._check_fp_sort(e)
        # Coral internally seems to use sun.misc.FloatingDecimal.readJavaFormatString()
        if self._is_float32_sort(e.sort()):
            self.sio.write("FCONST(Infinity)")
        elif self._is_float64_sort(e.sort()):
            self.sio.write("DCONST(Infinity)")
        else:
            raise CoralPrinterException("Unhandled +inf")

    def visit_float_minus_inf(self, e):
        assert e.num_args() == 0
        self._check_fp_sort(e)
        # Coral internally seems to use sun.misc.FloatingDecimal.readJavaFormatString()
        if self._is_float32_sort(e.sort()):
            self.sio.write("FCONST(-Infinity)")
        elif self._is_float64_sort(e.sort()):
            self.sio.write("DCONST(-Infinity)")
        else:
            raise CoralPrinterException("Unhandled -inf")

    def visit_float_nan(self, e):
        assert e.num_args() == 0
        self._check_fp_sort(e)
        # Coral internally seems to use sun.misc.FloatingDecimal.readJavaFormatString()
        if self._is_float32_sort(e.sort()):
            self.sio.write("FCONST(NaN)")
        elif self._is_float64_sort(e.sort()):
            self.sio.write("DCONST(NaN)")
        else:
            raise CoralPrinterException("Unhandled NaN")

    def visit_float_constant(self, e):
        assert e.num_args() == 3
        self._check_fp_sort(e)

        sign_bit = e.arg(0)
        assert isinstance(sign_bit, z3.BitVecNumRef)
        assert sign_bit.size() == 1
        sign_bit_bn = sign_bit.as_long()

        exp_bits = e.arg(1)
        assert isinstance(exp_bits, z3.BitVecNumRef)
        exp_bits_bn = exp_bits.as_long()

        # Excludes implicit bit
        significand_bits = e.arg(2)
        assert isinstance(significand_bits, z3.BitVecNumRef)
        significand_bits_bn = significand_bits.as_long()

        # Try to emit a hexfloat

        if exp_bits.size() == 8 and significand_bits.size() == 23:
            # Float 32
            self.sio.write("FCONST(")
            if exp_bits_bn == 0xFF:
                # Infinity or NaN
                if significand_bits_bn == 0:
                    # Infinity
                    if sign_bit_bn == 1:
                        self.sio.write("-Infinity)")
                    else:
                        self.sio.write("Infinity)")
                    return
                # NaN
                self.sio.write("NaN)")
                return

            # Normal or subnormal number
            if sign_bit_bn == 1:
                self.sio.write("-")

            # Zero
            if exp_bits_bn == 0 and significand_bits_bn == 0:
                self.sio.write("0.0)")
                return

            self.sio.write("0x")
            # Infer integer bit of signficand from exponent
            is_subnormal_or_zero = exp_bits_bn == 0
            if is_subnormal_or_zero:
                self.sio.write("0.")
            else:
                self.sio.write("1.")

            # Write out signficand out in hex
            # NOTE: need 6 hex digits
            significand_as_hex_str = f"{significand_bits_bn:06x}"
            assert len(significand_as_hex_str) == 6
            self.sio.write(significand_as_hex_str)

            # Now write out exponent
            normalized_exponent = exp_bits_bn - 127
            if normalized_exponent == -127:
                # Subnormal
                normalized_exponent = -126
            self.sio.write("p{}".format(normalized_exponent))
            self.sio.write(")")
        elif exp_bits.size() == 11 and significand_bits.size() == 52:
            # Float 64
            self.sio.write("DCONST(")
            if exp_bits_bn == 0x7FF:
                # Infinity or NaN
                if significand_bits_bn == 0:
                    # Infinity
                    if sign_bit_bn == 1:
                        self.sio.write("-Infinity)")
                    else:
                        self.sio.write("Infinity)")
                    return
                # NaN
                self.sio.write("NaN)")
                return

            # Normal or subnormal number
            if sign_bit_bn == 1:
                self.sio.write("-")

            # Zero
            if exp_bits_bn == 0 and significand_bits_bn == 0:
                self.sio.write("0.0)")
                return

            self.sio.write("0x")
            # Infer integer bit of signficand from exponent
            is_subnormal_or_zero = exp_bits_bn == 0
            if is_subnormal_or_zero:
                self.sio.write("0.")
            else:
                self.sio.write("1.")

            # Write out signficand out in hex
            # NOTE: need 13 hex digits
            significand_as_hex_str = f"{significand_bits_bn:013x}"
            assert len(significand_as_hex_str) == 13
            self.sio.write(significand_as_hex_str)

            # Now write out exponent
            normalized_exponent = exp_bits_bn - 1023
            if normalized_exponent == -1023:
                # Subnormal
                normalized_exponent = -1022
            self.sio.write("p{}".format(normalized_exponent))
            self.sio.write(")")
        else:
            raise CoralPrinterUnsupportedSort(e.sort())

    def _visit_binary_float_op(self, e, float32_name, float64_name):
        assert e.num_args() == 2
        self._check_fp_sort(e.arg(0))
        arg_sort = e.arg(0).sort()
        if self._is_float32_sort(arg_sort):
            self._visit_binary_op(e, float32_name)
        elif self._is_float64_sort(arg_sort):
            self._visit_binary_op(e, float64_name)
        else:
            raise CoralPrinterException("Unhandled binary float op case")

    def visit_float_eq(self, e):
        # FIXME: Check that these are semantically equivalent. This is tricky
        # because Coral is very poorly documented.
        self._visit_binary_float_op(e, "FEQ", "DEQ")

    def visit_float_leq(self, e):
        self._visit_binary_float_op(e, "FLE", "DLE")

    def visit_float_lt(self, e):
        self._visit_binary_float_op(e, "FLT", "DLT")

    def visit_float_geq(self, e):
        self._visit_binary_float_op(e, "FGE", "DGE")

    def visit_float_gt(self, e):
        self._visit_binary_float_op(e, "FGT", "DGT")

    def visit_float_neg(self, e):
        """
        Coral doesn't have this operation so we have to do something
        that is equivalent almost all the time

        (declare-const a Float32)
        (assert
          (not
            (=
              (fp.sub RNE (_ +zero 8 24) a)
              (fp.neg a)
            )
          )
        )
        (assert
          (not (= a (_ +zero 8 24)))
        )

        The above is unsat, i.e. apart from +zero (which coral doesn't support anyway)
        this transformation is sound.

        FIXME: Can we do any better?
        """
        self._unsound_translation("fp.neg")
        assert e.num_args() == 1
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = e.arg(0).sort()
        if self._is_float32_sort(arg_sort):
            self.sio.write("SUB(FCONST(0.0),")
            self.visit(arg)
            self.sio.write(")")
        elif self._is_float64_sort(arg_sort):
            self.sio.write("SUB(DCONST(0.0),")
            self.visit(arg)
            self.sio.write(")")
        else:
            raise CoralPrinterException("Unhandled fneg op case")

    def visit_float_abs(self, e):
        # FIXME: We need an ite expressions in coral to support this
        raise CoralPrinterUnsupportedOperation("fp.abs")

    def _check_rounding_mode(self, rounding_mode):
        assert isinstance(rounding_mode, z3.FPRMRef)
        kind = rounding_mode.decl().kind()
        # FIXME: Verify this assumption. RNE is usually the default
        # so Coral probably uses that...
        if kind == z3.Z3_OP_FPA_RM_NEAREST_TIES_TO_EVEN:
            return
        raise CoralPrinterUnsupportedRoundingMode(rounding_mode.as_string())

    def _visit_binary_arith_op_with_rounding_mode(self, e, op_name):
        assert e.num_args() == 3
        self._check_fp_sort(e)
        rounding_mode = e.arg(0)
        self._check_rounding_mode(rounding_mode)
        lhs = e.arg(1)
        rhs = e.arg(2)
        self.sio.write(op_name + "(")
        self.visit(lhs)
        self.sio.write(",")
        self.visit(rhs)
        self.sio.write(")")

    def visit_float_add(self, e):
        self._visit_binary_arith_op_with_rounding_mode(e, "ADD")

    def visit_float_sub(self, e):
        self._visit_binary_arith_op_with_rounding_mode(e, "SUB")

    def visit_float_mul(self, e):
        self._visit_binary_arith_op_with_rounding_mode(e, "MUL")

    def visit_float_div(self, e):
        self._visit_binary_arith_op_with_rounding_mode(e, "DIV")

    def visit_float_fma(self, e):
        raise CoralPrinterUnsupportedOperation("fp.fma")

    def visit_float_sqrt(self, e):
        assert e.num_args() == 2
        self._check_fp_sort(e)
        rounding_mode = e.arg(0)
        self._check_rounding_mode(rounding_mode)
        self.sio.write("SQRT_(")
        self.visit(e.arg(1))
        self.sio.write(")")

    def visit_float_rem(self, e):
        # FIXME: Check the semantics are correct here
        assert e.num_args() == 2
        self._check_fp_sort(e)
        self.sio.write("MOD(")
        self.visit(e.arg(0))
        self.sio.write(",")
        self.visit(e.arg(1))
        self.sio.write(")")

    def visit_float_round_to_integral(self, e):
        raise CoralPrinterUnsupportedOperation("fp.roundToIntegral")

    def visit_float_min(self, e):
        raise CoralPrinterUnsupportedOperation("fp.min")

    def visit_float_max(self, e):
        raise CoralPrinterUnsupportedOperation("fp.max")

    def visit_float_is_nan(self, e):
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = e.arg(0).sort()
        # (fp.isNaN a) <==>  (not (fp.eq a a))
        if self._is_float32_sort(arg_sort):
            self.sio.write("FNE(")
            self.visit(arg)
            self.sio.write(",")
            self.visit(arg)
            self.sio.write(")")
        elif self._is_float64_sort(arg_sort):
            self.sio.write("DNE(")
            self.visit(arg)
            self.sio.write(",")
            self.visit(arg)
            self.sio.write(")")
        else:
            raise CoralPrinterException("Unhandled fp.isNaN case")

    def visit_float_is_negative(self, e):
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = e.arg(0).sort()
        zero = None
        # FIXME: This isn't sound. We can't distinguish +0 and -0
        # in Coral's constraint language
        self._unsound_translation("fp.isNegative")
        if self._is_float32_sort(arg_sort):
            zero = z3.fpPlusZero(z3.Float32())
        elif self._is_float64_sort(arg_sort):
            zero = z3.fpPlusZero(z3.Float64())
        else:
            raise CoralPrinterException("Unhandled fp.isNegative case")
        tmp = z3.fpLT(arg, zero)
        self.visit(tmp)

    def visit_float_is_positive(self, e):
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = e.arg(0).sort()
        zero = None
        # FIXME: This isn't sound. We can't distinguish +0 and -0
        # in Coral's constraint language
        self._unsound_translation("fp.isPositive")
        if self._is_float32_sort(arg_sort):
            zero = z3.fpPlusZero(z3.Float32())
        elif self._is_float64_sort(arg_sort):
            zero = z3.fpPlusZero(z3.Float64())
        else:
            raise CoralPrinterException("Unhandled fp.isPositive case")
        tmp = z3.fpGT(arg, zero)
        self.visit(tmp)

    def _get_smallest_positive_normal_for(self, sort):
        if self._is_float32_sort(sort):
            return z3.fpFP(
                sgn=z3.BitVecVal(0, 1),
                exp=z3.BitVecVal(0x1, 8),
                sig=z3.BitVecVal(0x0, 23),
            )
        if self._is_float64_sort(sort):
            return z3.fpFP(
                sgn=z3.BitVecVal(0, 1),
                exp=z3.BitVecVal(0x1, 11),
                sig=z3.BitVecVal(0x0, 52),
            )
        raise CoralPrinterUnsupportedSort(sort)

    def _get_largest_negative_normal_for(self, sort):
        if self._is_float32_sort(sort):
            return z3.fpFP(
                sgn=z3.BitVecVal(1, 1),
                exp=z3.BitVecVal(0x1, 8),
                sig=z3.BitVecVal(0x0, 23),
            )
        if self._is_float64_sort(sort):
            return z3.fpFP(
                sgn=z3.BitVecVal(1, 1),
                exp=z3.BitVecVal(0x1, 11),
                sig=z3.BitVecVal(0x0, 52),
            )
        raise CoralPrinterUnsupportedSort(sort)

    def visit_float_is_normal(self, e):
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = e.arg(0).sort()
        smallest_positive_normal = self._get_smallest_positive_normal_for(arg_sort)
        largest_negative_normal = self._get_largest_negative_normal_for(arg_sort)
        temp = z3.Or(
            z3.And(
                z3.fpGEQ(arg, smallest_positive_normal),
                z3.fpLT(arg, z3.fpPlusInfinity(arg_sort)),
            ),
            z3.And(
                z3.fpLEQ(arg, largest_negative_normal),
                z3.fpGT(arg, z3.fpMinusInfinity(arg_sort)),
            ),
        )
        self.visit(temp)

    def visit_float_is_subnormal(self, e):
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = e.arg(0).sort()
        smallest_positive_normal = self._get_smallest_positive_normal_for(arg_sort)
        largest_negative_normal = self._get_largest_negative_normal_for(arg_sort)
        temp = z3.Or(
            z3.And(
                z3.fpLT(arg, smallest_positive_normal),
                z3.fpGT(arg, z3.fpPlusZero(arg_sort)),
            ),
            z3.And(
                z3.fpGT(arg, largest_negative_normal),
                z3.fpLT(arg, z3.fpMinusZero(arg_sort)),
            ),
        )
        self.visit(temp)

    def visit_float_is_zero(self, e):
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = arg.sort()
        zero = None
        # It doesn't matter if we pick +0 or -0 as we are using
        # the fp.eq operator which can't distinguish them, so
        # the choice of +0 is arbitrary.
        if self._is_float32_sort(arg_sort):
            zero = z3.fpPlusZero(z3.Float32())
        elif self._is_float64_sort(arg_sort):
            zero = z3.fpPlusZero(z3.Float64())
        else:
            raise CoralPrinterException("Unhandled fp.isZero op case")
        temp = z3.fpEQ(arg, zero)
        self.visit(temp)

    def visit_float_is_infinite(self, e):
        arg = e.arg(0)
        self._check_fp_sort(arg)
        arg_sort = arg.sort()
        pos_inf = None
        neg_inf = None
        if self._is_float32_sort(arg_sort):
            pos_inf = z3.fpPlusInfinity(z3.Float32())
            neg_inf = z3.fpMinusInfinity(z3.Float32())
        elif self._is_float64_sort(arg_sort):
            pos_inf = z3.fpPlusInfinity(z3.Float64())
            neg_inf = z3.fpMinusInfinity(z3.Float64())
        else:
            raise CoralPrinterException("Unhandled fp.isInfinite op case")
        temp = z3.Or(z3.fpEQ(arg, pos_inf), z3.fpEQ(arg, neg_inf))
        self.visit(temp)

    def visit_to_float(self, e):
        to_sort = e.sort()
        if e.num_args() == 1:
            from_sort = e.arg(0).sort()
        elif e.num_args() == 2:
            # First arg is rounding mode
            from_sort = e.arg(1).sort()
        else:
            from_sort = None
        # FIXME: Coral does support conversion between "int" and "double"
        # but doesn't seem to support anything else. For now just report
        # that this isn't supported.
        raise CoralPrinterUnsupportedOperation(f"Converting {from_sort} to {to_sort}")
