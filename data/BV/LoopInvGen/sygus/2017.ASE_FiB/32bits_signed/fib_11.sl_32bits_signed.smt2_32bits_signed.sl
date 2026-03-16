(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(i (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( bvsgt x #x00000000 ) ( = i #x00000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (i! (_ BitVec 32)) (x! (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt i x ) ( = j! ( bvadd j #x00000002 ) ) ( = i! ( bvadd i #x00000001 ) ) ( = x! x ) ) ( and ( bvsge i x ) ( = j! j ) ( = i! i ) ( = x! x ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle x i ) ) ( = j ( bvmul #x00000002 x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

