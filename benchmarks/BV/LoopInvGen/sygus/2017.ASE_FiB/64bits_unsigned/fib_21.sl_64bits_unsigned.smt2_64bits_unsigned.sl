(set-logic BV)

(synth-inv inv_fun ((v (_ BitVec 64))(n (_ BitVec 64))(c2 (_ BitVec 64))(c1 (_ BitVec 64))(k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((v (_ BitVec 64)) (n (_ BitVec 64)) (c2 (_ BitVec 64)) (c1 (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvugt n #x0000000000000000 ) ( bvult n #x000000000000000a ) ( = k #x0000000000000000 ) ( = i #x0000000000000000 ) ( = c1 #x0000000000000fa0 ) ( = c2 #x00000000000007d0 ) ))
(define-fun trans_fun ((v! (_ BitVec 64)) (n! (_ BitVec 64)) (c2! (_ BitVec 64)) (c1! (_ BitVec 64)) (k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (v (_ BitVec 64)) (n (_ BitVec 64)) (c2 (_ BitVec 64)) (c1 (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvuge i n ) ( = i! i ) ( = j! j ) ( = k! k ) ( = c1! c1 ) ( = c2! c2 ) ( = n! n ) ( = v! v ) ) ( and ( bvult i n ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! j ) ( = k! ( bvadd k c1 ) ) ( = c1! c1 ) ( = c2! c2 ) ( = n! n ) ( = v! #x0000000000000000 ) ) ( and ( bvult i n ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! j ) ( = k! ( bvadd k c2 ) ) ( = c1! c1 ) ( = c2! c2 ) ( = n! n ) ( = v! #x0000000000000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((v (_ BitVec 64)) (n (_ BitVec 64)) (c2 (_ BitVec 64)) (c1 (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvule k n ) ) ( not ( bvule n i ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

