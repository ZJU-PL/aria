(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 64))(i (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( bvsgt x #x0000000000000000 ) ( = i #x0000000000000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 64)) (i! (_ BitVec 64)) (x! (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt i x ) ( = j! ( bvadd j #x0000000000000002 ) ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = x! x ) ) ( and ( bvsge i x ) ( = j! j ) ( = i! i ) ( = x! x ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( not ( bvsle x i ) ) ( = j ( bvmul #x0000000000000002 x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

