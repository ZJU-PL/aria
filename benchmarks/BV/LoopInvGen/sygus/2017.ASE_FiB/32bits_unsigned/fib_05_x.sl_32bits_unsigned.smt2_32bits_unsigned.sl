(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(i (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ( = j #x00000000 ) ( = i #x00000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (i! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( = j! ( bvadd y! j ) ) ( = j! ( bvadd ( bvadd y! j ) #x00000001 ) ) ) ) ) ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = i! ( bvadd x! i ) ) a!1 ) ))
(define-fun post_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( bvule i j ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

