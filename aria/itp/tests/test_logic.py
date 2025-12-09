import aria.itp   as kd
import aria.itp.smt as smt
import aria.itp.theories.logic as logic
import aria.itp.theories.logic.zf as zf
import aria.itp.theories.logic.robinson as robinson
import aria.itp.theories.logic.peano as peano
import pytest
import aria.itp.theories.logic.intuitionistic as intuit

def test_logic():
    assert True


def test_intuit():
    a = smt.Const("a", intuit.Prop)
    with pytest.raises(kd.kernel.LemmaError):
        excluded_middle = kd.prove(kd.QForAll([a], intuit.Valid(a | ~a)), by=[intuit.acc_refl, intuit.acc_trans])
    with pytest.raises(kd.kernel.LemmaError):
        dne = kd.prove(kd.QForAll([a], intuit.Valid(intuit.Implies(~~a, a))), by=[intuit.acc_refl, intuit.acc_trans])
