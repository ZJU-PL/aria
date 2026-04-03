(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvsgt x ( bvsub #x0000000000000000 #x0000000000000064 ) ) ( bvslt x #x00000000000000c8 ) ( bvsgt z #x0000000000000064 ) ( bvslt z #x00000000000000c8 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = z! z ) ) ( and ( = x! ( bvsub x #x0000000000000001 ) ) ( = z! ( bvsub z #x0000000000000001 ) ) ) ) ) ) ( and ( bvslt x #x0000000000000064 ) ( bvsgt z #x0000000000000064 ) a!1 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( and ( not ( bvsle #x0000000000000064 x ) ) ( not ( bvsle z #x0000000000000064 ) ) ) ( bvsle z #x0000000000000064 ) ( bvsle #x0000000000000064 x ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

