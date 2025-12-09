import aria.itp as itp
import aria.itp.smt as smt

Ob = smt.DeclareSort("Ob")
a, b, c, d = smt.Consts("a b c d", Ob)

Arr = smt.DeclareSort("Arr")
f, g, h, k = smt.Consts("f g h k", Arr)

dom = smt.Function("dom", Arr, Ob)
cod = smt.Function("cod", Arr, Ob)

# not all arrow expressions are well formed.
wf = smt.Function("wf", Arr, smt.BoolSort())
itp.notation.wf.register(Arr, wf)

comp = smt.Function("comp", Arr, Arr, Arr)
itp.notation.matmul.register(Arr, comp)
wf_comp = itp.axiom(itp.QForAll([f, g], cod(g) == dom(f), wf(f @ g)))

id = smt.Function("id", Ob, Arr)
wf_id = itp.axiom(smt.ForAll([a], wf(id(a))))
