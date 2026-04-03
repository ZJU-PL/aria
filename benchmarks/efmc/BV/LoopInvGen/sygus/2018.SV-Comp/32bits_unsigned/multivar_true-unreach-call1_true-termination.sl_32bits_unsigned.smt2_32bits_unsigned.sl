(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = y x ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvult x #x00000400 ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvule #x00000400 x ) ) ( = x y ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

