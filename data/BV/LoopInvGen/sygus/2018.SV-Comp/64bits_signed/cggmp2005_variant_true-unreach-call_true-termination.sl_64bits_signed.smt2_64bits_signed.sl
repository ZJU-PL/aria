(set-logic BV)

(synth-inv inv_fun ((hi (_ BitVec 64))(mid (_ BitVec 64))(lo (_ BitVec 64))))

(define-fun pre_fun ((hi (_ BitVec 64)) (mid (_ BitVec 64)) (lo (_ BitVec 64))) Bool
       ( and ( = lo #x0000000000000000 ) ( bvsgt mid #x0000000000000000 ) ( = hi ( bvmul #x0000000000000002 mid ) ) ))
(define-fun trans_fun ((hi! (_ BitVec 64)) (mid! (_ BitVec 64)) (lo! (_ BitVec 64)) (hi (_ BitVec 64)) (mid (_ BitVec 64)) (lo (_ BitVec 64))) Bool
       ( and ( bvsgt mid #x0000000000000000 ) ( = lo! ( bvadd lo #x0000000000000001 ) ) ( = hi! ( bvsub hi #x0000000000000001 ) ) ( = mid! ( bvsub mid #x0000000000000001 ) ) ))
(define-fun post_fun ((hi (_ BitVec 64)) (mid (_ BitVec 64)) (lo (_ BitVec 64))) Bool
       ( or ( not ( bvsle mid #x0000000000000000 ) ) ( = lo hi ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

