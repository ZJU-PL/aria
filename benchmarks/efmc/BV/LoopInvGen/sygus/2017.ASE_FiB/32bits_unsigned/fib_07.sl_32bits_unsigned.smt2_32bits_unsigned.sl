(set-logic BV)

(synth-inv inv_fun ((b (_ BitVec 32))(a (_ BitVec 32))(n (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((b (_ BitVec 32)) (a (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = a #x00000000 ) ( = b #x00000000 ) ( bvuge n #x00000000 ) ))
(define-fun trans_fun ((b! (_ BitVec 32)) (a! (_ BitVec 32)) (n! (_ BitVec 32)) (i! (_ BitVec 32)) (b (_ BitVec 32)) (a (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvult i n ) ( = i! ( bvadd i #x00000001 ) ) ( = a! ( bvadd a #x00000001 ) ) ( = b! ( bvadd b #x00000002 ) ) ( = n! n ) ) ( and ( bvult i n ) ( = i! ( bvadd i #x00000001 ) ) ( = a! ( bvadd a #x00000002 ) ) ( = b! ( bvadd b #x00000001 ) ) ( = n! n ) ) ( and ( bvuge i n ) ( = i! i ) ( = a! a ) ( = b! b ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((b (_ BitVec 32)) (a (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvule n i ) ) ( = ( bvadd a b ) ( bvmul #x00000003 n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

