(set-logic BV)

(synth-inv inv_fun ((j1 (_ BitVec 64))(i1 (_ BitVec 64))(k (_ BitVec 64))(i0 (_ BitVec 64))(n1 (_ BitVec 64))(n0 (_ BitVec 64))))

(define-fun pre_fun ((j1 (_ BitVec 64)) (i1 (_ BitVec 64)) (k (_ BitVec 64)) (i0 (_ BitVec 64)) (n1 (_ BitVec 64)) (n0 (_ BitVec 64))) Bool
       ( and ( bvsle ( bvsub #x0000000000000000 #x00000000000f4240 ) n0 ) ( bvslt n0 #x00000000000f4240 ) ( bvsle ( bvsub #x0000000000000000 #x00000000000f4240 ) n1 ) ( bvslt n1 #x00000000000f4240 ) ( = i1 #x0000000000000000 ) ( = j1 #x0000000000000000 ) ( = i0 #x0000000000000000 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((j1! (_ BitVec 64)) (i1! (_ BitVec 64)) (k! (_ BitVec 64)) (i0! (_ BitVec 64)) (n1! (_ BitVec 64)) (n0! (_ BitVec 64)) (j1 (_ BitVec 64)) (i1 (_ BitVec 64)) (k (_ BitVec 64)) (i0 (_ BitVec 64)) (n1 (_ BitVec 64)) (n0 (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt i0 n0 ) ( = i0! ( bvadd i0 #x0000000000000001 ) ) ( = k! ( bvadd k #x0000000000000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i1! i1 ) ( = j1! j1 ) ) ( and ( bvsge i0 n0 ) ( bvslt i1 n1 ) ( = i1! ( bvadd i1 #x0000000000000001 ) ) ( = k! ( bvadd k #x0000000000000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i0! i0 ) ( = j1! j1 ) ) ( and ( bvsge i0 n0 ) ( bvsge i1 n1 ) ( bvslt j1 ( bvadd n0 n1 ) ) ( = j1! ( bvadd j1 #x0000000000000001 ) ) ( = k! ( bvsub k #x0000000000000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i0! i0 ) ( = i1! i1 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j1 (_ BitVec 64)) (i1 (_ BitVec 64)) (k (_ BitVec 64)) (i0 (_ BitVec 64)) (n1 (_ BitVec 64)) (n0 (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvsle n0 i0 ) ( bvsle n1 i1 ) ( not ( bvsle ( bvadd n0 n1 ) j1 ) ) ) ) ) ( or ( not a!1 ) ( not ( bvsle k #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

