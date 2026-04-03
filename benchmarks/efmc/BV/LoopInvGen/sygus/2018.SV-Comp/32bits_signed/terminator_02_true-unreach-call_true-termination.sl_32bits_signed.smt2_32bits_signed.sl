(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvsgt x ( bvsub #x00000000 #x00000064 ) ) ( bvslt x #x000000c8 ) ( bvsgt z #x00000064 ) ( bvslt z #x000000c8 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x00000001 ) ) ( = z! z ) ) ( and ( = x! ( bvsub x #x00000001 ) ) ( = z! ( bvsub z #x00000001 ) ) ) ) ) ) ( and ( bvslt x #x00000064 ) ( bvsgt z #x00000064 ) a!1 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( bvsle z #x00000064 ) ( and ( not ( bvsle #x00000064 x ) ) ( not ( bvsle z #x00000064 ) ) ) ( bvsle #x00000064 x ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

