(set-logic BV)

(synth-inv inv_fun ((cnt (_ BitVec 64))(res (_ BitVec 64))(b (_ BitVec 64))(a (_ BitVec 64))))

(define-fun pre_fun ((cnt (_ BitVec 64)) (res (_ BitVec 64)) (b (_ BitVec 64)) (a (_ BitVec 64))) Bool
       ( and ( bvsle a #x00000000000f4240 ) ( bvsle #x0000000000000000 b ) ( bvsle b #x00000000000f4240 ) ( = res a ) ( = cnt b ) ))
(define-fun trans_fun ((cnt! (_ BitVec 64)) (res! (_ BitVec 64)) (b! (_ BitVec 64)) (a! (_ BitVec 64)) (cnt (_ BitVec 64)) (res (_ BitVec 64)) (b (_ BitVec 64)) (a (_ BitVec 64))) Bool
       ( and ( bvsgt cnt #x0000000000000000 ) ( = cnt! ( bvsub cnt #x0000000000000001 ) ) ( = res! ( bvadd res #x0000000000000001 ) ) ( = a! a ) ( = b! b ) ))
(define-fun post_fun ((cnt (_ BitVec 64)) (res (_ BitVec 64)) (b (_ BitVec 64)) (a (_ BitVec 64))) Bool
       ( or ( not ( bvsle cnt #x0000000000000000 ) ) ( = res ( bvadd a b ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

