(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 64))(i (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ( = j #x0000000000000000 ) ( = i #x0000000000000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 64)) (i! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( = j! ( bvadd y! j ) ) ( = j! ( bvadd ( bvadd y! j ) #x0000000000000001 ) ) ) ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = i! ( bvadd x! i ) ) a!1 ) ))
(define-fun post_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( bvsle i j ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

