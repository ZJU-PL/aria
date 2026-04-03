(set-logic BV)

(synth-inv inv_fun ((j1 (_ BitVec 64))(i1 (_ BitVec 64))(k (_ BitVec 64))(i0 (_ BitVec 64))(n1 (_ BitVec 64))(n0 (_ BitVec 64))))

(define-fun pre_fun ((j1 (_ BitVec 64)) (i1 (_ BitVec 64)) (k (_ BitVec 64)) (i0 (_ BitVec 64)) (n1 (_ BitVec 64)) (n0 (_ BitVec 64))) Bool
       ( and ( bvule ( bvsub #x0000000000000000 #x00000000000f4240 ) n0 ) ( bvult n0 #x00000000000f4240 ) ( bvule ( bvsub #x0000000000000000 #x00000000000f4240 ) n1 ) ( bvult n1 #x00000000000f4240 ) ( = i1 #x0000000000000000 ) ( = j1 #x0000000000000000 ) ( = i0 #x0000000000000000 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((j1! (_ BitVec 64)) (i1! (_ BitVec 64)) (k! (_ BitVec 64)) (i0! (_ BitVec 64)) (n1! (_ BitVec 64)) (n0! (_ BitVec 64)) (j1 (_ BitVec 64)) (i1 (_ BitVec 64)) (k (_ BitVec 64)) (i0 (_ BitVec 64)) (n1 (_ BitVec 64)) (n0 (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvult i0 n0 ) ( = i0! ( bvadd i0 #x0000000000000001 ) ) ( = k! ( bvadd k #x0000000000000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i1! i1 ) ( = j1! j1 ) ) ( and ( bvuge i0 n0 ) ( bvult i1 n1 ) ( = i1! ( bvadd i1 #x0000000000000001 ) ) ( = k! ( bvadd k #x0000000000000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i0! i0 ) ( = j1! j1 ) ) ( and ( bvuge i0 n0 ) ( bvuge i1 n1 ) ( bvult j1 ( bvadd n0 n1 ) ) ( = j1! ( bvadd j1 #x0000000000000001 ) ) ( = k! ( bvsub k #x0000000000000001 ) ) ( = n0! n0 ) ( = n1! n1 ) ( = i0! i0 ) ( = i1! i1 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j1 (_ BitVec 64)) (i1 (_ BitVec 64)) (k (_ BitVec 64)) (i0 (_ BitVec 64)) (n1 (_ BitVec 64)) (n0 (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvule n0 i0 ) ( bvule n1 i1 ) ( not ( bvule ( bvadd n0 n1 ) j1 ) ) ) ) ) ( or ( not a!1 ) ( not ( = k #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

