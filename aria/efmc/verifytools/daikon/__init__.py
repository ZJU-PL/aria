"""Daikon invariant detection integration."""

from os.path import basename
from subprocess import call, check_output
from tempfile import NamedTemporaryFile

from aria.efmc.verifytools.common.util import (
    error,
    flattenList,
)  # pylint: disable=no-name-in-module
from aria.efmc.verifytools.daikon.inv_ast import parse_expr_ast


def to_daikon_trace(varz, trace):
    """Convert variables and trace data to Daikon trace format."""
    r = "decl-version 2.0\n"
    r += "ppt LoopEntry:::\n"
    r += "ppt-type point\n"
    for v in varz:
        r += "  variable " + v + "\n"
        r += "    var-kind variable\n"
        r += "    dec-type int\n"
        r += "    rep-type int\n"

    r += "\n"
    for row in trace:
        r += "LoopEntry:::\n"
        for var, val in zip(varz, row):
            r += var + "\n" + str(val) + "\n1\n"
        r += "\n"
    return r


inv_enable_opts = [
    "daikon.inv.binary.sequenceScalar.Member.enabled",
    "daikon.inv.binary.sequenceScalar.MemberFloat.enabled",
    "daikon.inv.binary.sequenceScalar.SeqFloatEqual.enabled",
    "daikon.inv.binary.sequenceScalar.SeqFloatGreaterEqual.enabled",
    "daikon.inv.binary.sequenceScalar.SeqFloatGreaterThan.enabled",
    "daikon.inv.binary.sequenceScalar.SeqFloatLessEqual.enabled",
    "daikon.inv.binary.sequenceScalar.SeqFloatLessThan.enabled",
    "daikon.inv.binary.sequenceScalar.SeqIntEqual.enabled",
    "daikon.inv.binary.sequenceScalar.SeqIntGreaterEqual.enabled",
    "daikon.inv.binary.sequenceScalar.SeqIntGreaterThan.enabled",
    "daikon.inv.binary.sequenceScalar.SeqIntLessEqual.enabled",
    "daikon.inv.binary.sequenceScalar.SeqIntLessThan.enabled",
    "daikon.inv.binary.sequenceString.MemberString.enabled",
    "daikon.inv.binary.twoScalar.FloatEqual.enabled",
    "daikon.inv.binary.twoScalar.FloatGreaterEqual.enabled",
    "daikon.inv.binary.twoScalar.FloatGreaterThan.enabled",
    "daikon.inv.binary.twoScalar.FloatLessEqual.enabled",
    "daikon.inv.binary.twoScalar.FloatLessThan.enabled",
    "daikon.inv.binary.twoScalar.FloatNonEqual.enabled",
    "daikon.inv.binary.twoScalar.IntEqual.enabled",
    "daikon.inv.binary.twoScalar.IntGreaterEqual.enabled",
    "daikon.inv.binary.twoScalar.IntGreaterThan.enabled",
    "daikon.inv.binary.twoScalar.IntLessEqual.enabled",
    "daikon.inv.binary.twoScalar.IntLessThan.enabled",
    "daikon.inv.binary.twoScalar.IntNonEqual.enabled",
    "daikon.inv.binary.twoScalar.LinearBinary.enabled",
    "daikon.inv.binary.twoScalar.LinearBinaryFloat.enabled",
    "daikon.inv.binary.twoScalar.NumericFloat.Divides.enabled",
    "daikon.inv.binary.twoScalar.NumericFloat.Square.enabled",
    "daikon.inv.binary.twoScalar.NumericFloat.ZeroTrack.enabled",
    "daikon.inv.binary.twoScalar.NumericInt.BitwiseAndZero.enabled",
    "daikon.inv.binary.twoScalar.NumericInt.BitwiseComplement.enabled",
    "daikon.inv.binary.twoScalar.NumericInt.BitwiseSubset.enabled",
    "daikon.inv.binary.twoScalar.NumericInt.Divides.enabled",
    "daikon.inv.binary.twoScalar.NumericInt.ShiftZero.enabled",
    "daikon.inv.binary.twoScalar.NumericInt.Square.enabled",
    "daikon.inv.binary.twoScalar.NumericInt.ZeroTrack.enabled",
    "daikon.inv.binary.twoSequence.PairwiseFloatEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseFloatGreaterEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseFloatGreaterThan.enabled",
    "daikon.inv.binary.twoSequence.PairwiseFloatLessEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseFloatLessThan.enabled",
    "daikon.inv.binary.twoSequence.PairwiseIntEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseIntGreaterEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseIntGreaterThan.enabled",
    "daikon.inv.binary.twoSequence.PairwiseIntLessEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseIntLessThan.enabled",
    "daikon.inv.binary.twoSequence.PairwiseLinearBinary.enabled",
    "daikon.inv.binary.twoSequence.PairwiseLinearBinaryFloat.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericFloat.Divides.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericFloat.Square.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericFloat.ZeroTrack.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericInt.BitwiseAndZero.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericInt.BitwiseComplement.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericInt.BitwiseSubset.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericInt.Divides.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericInt.ShiftZero.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericInt.Square.enabled",
    "daikon.inv.binary.twoSequence.PairwiseNumericInt.ZeroTrack.enabled",
    "daikon.inv.binary.twoSequence.PairwiseString.SubString.enabled",
    "daikon.inv.binary.twoSequence.PairwiseStringEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseStringGreaterEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseStringGreaterThan.enabled",
    "daikon.inv.binary.twoSequence.PairwiseStringLessEqual.enabled",
    "daikon.inv.binary.twoSequence.PairwiseStringLessThan.enabled",
    "daikon.inv.binary.twoSequence.Reverse.enabled",
    "daikon.inv.binary.twoSequence.ReverseFloat.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqFloatEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqFloatGreaterEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqFloatGreaterThan.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqFloatLessEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqFloatLessThan.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqIntEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqIntGreaterEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqIntGreaterThan.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqIntLessEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqIntLessThan.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqStringEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqStringGreaterEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqStringGreaterThan.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqStringLessEqual.enabled",
    "daikon.inv.binary.twoSequence.SeqSeqStringLessThan.enabled",
    "daikon.inv.binary.twoSequence.SubSequence.enabled",
    "daikon.inv.binary.twoSequence.SubSequenceFloat.enabled",
    "daikon.inv.binary.twoSequence.SubSet.enabled",
    "daikon.inv.binary.twoSequence.SubSetFloat.enabled",
    "daikon.inv.binary.twoSequence.SuperSequence.enabled",
    "daikon.inv.binary.twoSequence.SuperSequenceFloat.enabled",
    "daikon.inv.binary.twoSequence.SuperSet.enabled",
    "daikon.inv.binary.twoSequence.SuperSetFloat.enabled",
    "daikon.inv.binary.twoString.StdString.SubString.enabled",
    "daikon.inv.binary.twoString.StringEqual.enabled",
    "daikon.inv.binary.twoString.StringGreaterEqual.enabled",
    "daikon.inv.binary.twoString.StringGreaterThan.enabled",
    "daikon.inv.binary.twoString.StringLessEqual.enabled",
    "daikon.inv.binary.twoString.StringLessThan.enabled",
    "daikon.inv.binary.twoString.StringNonEqual.enabled",
    "daikon.inv.ternary.threeScalar.FunctionBinary.enabled",
    "daikon.inv.ternary.threeScalar.FunctionBinaryFloat.enabled",
    "daikon.inv.ternary.threeScalar.LinearTernary.enabled",
    "daikon.inv.ternary.threeScalar.LinearTernaryFloat.enabled",
    "daikon.inv.unary.scalar.CompleteOneOfScalar.enabled",
    "daikon.inv.unary.scalar.IsPointer.enabled",
    "daikon.inv.unary.scalar.LowerBound.enabled",
    "daikon.inv.unary.scalar.LowerBoundFloat.enabled",
    "daikon.inv.unary.scalar.Modulus.enabled",
    "daikon.inv.unary.scalar.NonModulus.enabled",
    "daikon.inv.unary.scalar.NonZero.enabled",
    "daikon.inv.unary.scalar.NonZeroFloat.enabled",
    "daikon.inv.unary.scalar.OneOfFloat.enabled",
    "daikon.inv.unary.scalar.OneOfScalar.enabled",
    "daikon.inv.unary.scalar.Positive.enabled",
    "daikon.inv.unary.scalar.RangeInt.Even.enabled",
    "daikon.inv.unary.scalar.RangeInt.PowerOfTwo.enabled",
    "daikon.inv.unary.scalar.UpperBound.enabled",
    "daikon.inv.unary.scalar.UpperBoundFloat.enabled",
    "daikon.inv.unary.sequence.CommonFloatSequence.enabled",
    "daikon.inv.unary.sequence.CommonSequence.enabled",
    "daikon.inv.unary.sequence.EltLowerBound.enabled",
    "daikon.inv.unary.sequence.EltLowerBoundFloat.enabled",
    "daikon.inv.unary.sequence.EltNonZero.enabled",
    "daikon.inv.unary.sequence.EltNonZeroFloat.enabled",
    "daikon.inv.unary.sequence.EltOneOf.enabled",
    "daikon.inv.unary.sequence.EltOneOfFloat.enabled",
    "daikon.inv.unary.sequence.EltRangeInt.Even.enabled",
    "daikon.inv.unary.sequence.EltRangeInt.PowerOfTwo.enabled",
    "daikon.inv.unary.sequence.EltUpperBound.enabled",
    "daikon.inv.unary.sequence.EltUpperBoundFloat.enabled",
    "daikon.inv.unary.sequence.EltwiseFloatEqual.enabled",
    "daikon.inv.unary.sequence.EltwiseFloatGreaterEqual.enabled",
    "daikon.inv.unary.sequence.EltwiseFloatGreaterThan.enabled",
    "daikon.inv.unary.sequence.EltwiseFloatLessEqual.enabled",
    "daikon.inv.unary.sequence.EltwiseFloatLessThan.enabled",
    "daikon.inv.unary.sequence.EltwiseIntEqual.enabled",
    "daikon.inv.unary.sequence.EltwiseIntGreaterEqual.enabled",
    "daikon.inv.unary.sequence.EltwiseIntGreaterThan.enabled",
    "daikon.inv.unary.sequence.EltwiseIntLessEqual.enabled",
    "daikon.inv.unary.sequence.EltwiseIntLessThan.enabled",
    "daikon.inv.unary.sequence.NoDuplicates.enabled",
    "daikon.inv.unary.sequence.NoDuplicatesFloat.enabled",
    "daikon.inv.unary.sequence.OneOfFloatSequence.enabled",
    "daikon.inv.unary.sequence.OneOfSequence.enabled",
    "daikon.inv.unary.sequence.SeqIndexFloatEqual.enabled",
    "daikon.inv.unary.sequence.SeqIndexFloatGreaterEqual.enabled",
    "daikon.inv.unary.sequence.SeqIndexFloatGreaterThan.enabled",
    "daikon.inv.unary.sequence.SeqIndexFloatLessEqual.enabled",
    "daikon.inv.unary.sequence.SeqIndexFloatLessThan.enabled",
    "daikon.inv.unary.sequence.SeqIndexFloatNonEqual.enabled",
    "daikon.inv.unary.sequence.SeqIndexIntEqual.enabled",
    "daikon.inv.unary.sequence.SeqIndexIntGreaterEqual.enabled",
    "daikon.inv.unary.sequence.SeqIndexIntGreaterThan.enabled",
    "daikon.inv.unary.sequence.SeqIndexIntLessEqual.enabled",
    "daikon.inv.unary.sequence.SeqIndexIntLessThan.enabled",
    "daikon.inv.unary.sequence.SeqIndexIntNonEqual.enabled",
    "daikon.inv.unary.string.CompleteOneOfString.enabled",
    "daikon.inv.unary.string.OneOfString.enabled",
    "daikon.inv.unary.string.PrintableString.enabled",
    "daikon.inv.unary.stringsequence.CommonStringSequence.enabled",
    "daikon.inv.unary.stringsequence.EltOneOfString.enabled",
    "daikon.inv.unary.stringsequence.OneOfStringSequence.enabled",
]

filter_opts = [
    "daikon.inv.filter.ObviousFilter.enabled",
    "daikon.inv.filter.OnlyConstantVariablesFilter.enabled",
    "daikon.inv.filter.ParentFilter.enabled",
    "daikon.inv.filter.SimplifyFilter.enabled",
    "daikon.inv.filter.UnjustifiedFilter.enabled",
]


def run_daikon(varz, trace, nosuppress=False):
    """Run Daikon on the given variables and trace data."""
    with NamedTemporaryFile(suffix=".dtrace", delete=False) as dtrace_f:
        dtrace_f.write(to_daikon_trace(varz, trace))
        dtrace_f.flush()
        args = ["java", "daikon.Daikon"]
        if nosuppress:
            # Lets disable all filters and enable all invariants
            args = (
                args
                + flattenList([["--config_option", x + "=false"] for x in filter_opts])
                + flattenList(
                    [["--config_option", x + "=true"] for x in inv_enable_opts]
                )
            )
        args.append(dtrace_f.name)
        raw = check_output(args)
        call(["rm", basename(dtrace_f.name)[: -len(".dtrace")] + ".inv.gz"])
        start = raw.index("LoopEntry:::") + len("LoopEntry:::")
        end = raw.index("Exiting Daikon.", start)
        invs = filter(
            lambda x: x != "", map(lambda x: x.strip(), raw[start:end].split("\n"))
        )
        # I don't understand how LinearTeranry invariants without justification are
        # displayed... Also getting lazy to fix the long line..
        invs = filter(
            lambda x: "warning: too few samples for daikon.inv.ternary.threeScalar.LinearTernary invariant"
            not in x,
            invs,
        )  # pylint: disable=line-too-long
        invs = filter(lambda x: "(mod 0)" not in x, invs)
        parsable = []
        for i in invs:
            try:
                parsable.append(parse_expr_ast(i))
            except Exception:  # pylint: disable=broad-except
                error("Failed parsing: ", i)
        return parsable
