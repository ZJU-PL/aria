(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 64))(v2 (_ BitVec 64))(v1 (_ BitVec 64))(n (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x #x0000000000000001 ))
(define-fun trans_fun ((v3! (_ BitVec 64)) (v2! (_ BitVec 64)) (v1! (_ BitVec 64)) (n! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvsle x n ) ( = y! ( bvsub n x ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ))
(define-fun post_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvsle x n ) ( = y ( bvadd n ( bvmul #xffffffffffffffff x ) ) ) ( or ( not ( bvsle #x0000000000000000 y ) ) ( bvsle n y ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

