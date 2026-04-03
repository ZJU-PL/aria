(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x000000c8 ) ( = z #x00000190 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd #x00000001 x ) ) ( = y! ( ite ( bvslt x #x000000c8 ) ( bvadd y #x00000001 ) y ) ) ( = z! ( ite ( bvslt x #x000000c8 ) z ( bvadd z #x00000002 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( bvsle #x00000190 y ) ( = z ( bvmul #x00000002 x ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

