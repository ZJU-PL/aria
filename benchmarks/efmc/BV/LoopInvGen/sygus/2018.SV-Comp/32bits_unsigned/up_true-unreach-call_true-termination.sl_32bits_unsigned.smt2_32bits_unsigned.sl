(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(k (_ BitVec 32))(i (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = k #x00000000 ) ( = j #x00000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (k! (_ BitVec 32)) (i! (_ BitVec 32)) (n! (_ BitVec 32)) (j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvult i n ) ( = i! ( bvadd i #x00000001 ) ) ( = k! ( bvadd k #x00000001 ) ) ( = n! n ) ( = j! j ) ) ( and ( bvuge i n ) ( bvult j n ) ( = j! ( bvadd j #x00000001 ) ) ( = k! ( bvsub k #x00000001 ) ) ( = n! n ) ( = i! i ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( bvule n i ) ( not ( bvule n j ) ) ) ) ) ) ( or ( not ( = k #x00000000 ) ) a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

