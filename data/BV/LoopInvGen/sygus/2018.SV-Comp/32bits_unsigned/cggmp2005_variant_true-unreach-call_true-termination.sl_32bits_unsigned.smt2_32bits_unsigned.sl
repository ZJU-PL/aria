(set-logic BV)

(synth-inv inv_fun ((hi (_ BitVec 32))(mid (_ BitVec 32))(lo (_ BitVec 32))))

(define-fun pre_fun ((hi (_ BitVec 32)) (mid (_ BitVec 32)) (lo (_ BitVec 32))) Bool
       ( and ( = lo #x00000000 ) ( bvugt mid #x00000000 ) ( = hi ( bvmul #x00000002 mid ) ) ))
(define-fun trans_fun ((hi! (_ BitVec 32)) (mid! (_ BitVec 32)) (lo! (_ BitVec 32)) (hi (_ BitVec 32)) (mid (_ BitVec 32)) (lo (_ BitVec 32))) Bool
       ( and ( bvugt mid #x00000000 ) ( = lo! ( bvadd lo #x00000001 ) ) ( = hi! ( bvsub hi #x00000001 ) ) ( = mid! ( bvsub mid #x00000001 ) ) ))
(define-fun post_fun ((hi (_ BitVec 32)) (mid (_ BitVec 32)) (lo (_ BitVec 32))) Bool
       ( or ( not ( = mid #x00000000 ) ) ( = lo hi ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

