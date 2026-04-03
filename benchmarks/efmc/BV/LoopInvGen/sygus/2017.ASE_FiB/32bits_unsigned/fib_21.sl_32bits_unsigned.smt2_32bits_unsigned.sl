(set-logic BV)

(synth-inv inv_fun ((v (_ BitVec 32))(n (_ BitVec 32))(c2 (_ BitVec 32))(c1 (_ BitVec 32))(k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((v (_ BitVec 32)) (n (_ BitVec 32)) (c2 (_ BitVec 32)) (c1 (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvugt n #x00000000 ) ( bvult n #x0000000a ) ( = k #x00000000 ) ( = i #x00000000 ) ( = c1 #x00000fa0 ) ( = c2 #x000007d0 ) ))
(define-fun trans_fun ((v! (_ BitVec 32)) (n! (_ BitVec 32)) (c2! (_ BitVec 32)) (c1! (_ BitVec 32)) (k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (v (_ BitVec 32)) (n (_ BitVec 32)) (c2 (_ BitVec 32)) (c1 (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvuge i n ) ( = i! i ) ( = j! j ) ( = k! k ) ( = c1! c1 ) ( = c2! c2 ) ( = n! n ) ( = v! v ) ) ( and ( bvult i n ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = k! ( bvadd k c1 ) ) ( = c1! c1 ) ( = c2! c2 ) ( = n! n ) ( = v! #x00000000 ) ) ( and ( bvult i n ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = k! ( bvadd k c2 ) ) ( = c1! c1 ) ( = c2! c2 ) ( = n! n ) ( = v! #x00000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((v (_ BitVec 32)) (n (_ BitVec 32)) (c2 (_ BitVec 32)) (c1 (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvule k n ) ) ( not ( bvule n i ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

