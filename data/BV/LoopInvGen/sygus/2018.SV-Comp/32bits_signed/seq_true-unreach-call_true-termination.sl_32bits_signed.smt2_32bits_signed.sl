(set-logic BV)

(synth-inv inv_fun ((j1 (_ BitVec 32))(i1 (_ BitVec 32))(k (_ BitVec 32))(i0 (_ BitVec 32))(n1 (_ BitVec 32))(n0 (_ BitVec 32))))

(define-fun pre_fun ((j1 (_ BitVec 32)) (i1 (_ BitVec 32)) (k (_ BitVec 32)) (i0 (_ BitVec 32)) (n1 (_ BitVec 32)) (n0 (_ BitVec 32))) Bool
       ( and ( bvsle ( bvsub #x00000000 #x000f4240 ) n0 ) ( bvslt n0 #x000f4240 ) ( bvsle ( bvsub #x00000000 #x000f4240 ) n1 ) ( bvslt n1 #x000f4240 ) ( = i1 #x00000000 ) ( = j1 #x00000000 ) ( = i0 #x00000000 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((j1! (_ BitVec 32)) (i1! (_ BitVec 32)) (k! (_ BitVec 32)) (i0! (_ BitVec 32)) (n1! (_ BitVec 32)) (n0! (_ BitVec 32)) (j1 (_ BitVec 32)) (i1 (_ BitVec 32)) (k (_ BitVec 32)) (i0 (_ BitVec 32)) (n1 (_ BitVec 32)) (n0 (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt i0 n0 ) ( = i0! ( bvadd i0 #x00000001 ) ) ( = k! ( bvadd k #x00000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i1! i1 ) ( = j1! j1 ) ) ( and ( bvsge i0 n0 ) ( bvslt i1 n1 ) ( = i1! ( bvadd i1 #x00000001 ) ) ( = k! ( bvadd k #x00000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i0! i0 ) ( = j1! j1 ) ) ( and ( bvsge i0 n0 ) ( bvsge i1 n1 ) ( bvslt j1 ( bvadd n0 n1 ) ) ( = j1! ( bvadd j1 #x00000001 ) ) ( = k! ( bvsub k #x00000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i0! i0 ) ( = i1! i1 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j1 (_ BitVec 32)) (i1 (_ BitVec 32)) (k (_ BitVec 32)) (i0 (_ BitVec 32)) (n1 (_ BitVec 32)) (n0 (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( bvsle n0 i0 ) ( bvsle n1 i1 ) ( not ( bvsle ( bvadd n0 n1 ) j1 ) ) ) ) ) ( or ( not a!1 ) ( not ( bvsle k #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

