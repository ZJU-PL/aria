(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 32))(v2 (_ BitVec 32))(v1 (_ BitVec 32))(n (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000001 ))
(define-fun trans_fun ((v3! (_ BitVec 32)) (v2! (_ BitVec 32)) (v1! (_ BitVec 32)) (n! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvsle x n ) ( = y! ( bvsub n x ) ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( bvsle x n ) ( = y ( bvadd n ( bvmul #xffffffff x ) ) ) ( or ( bvsle n y ) ( not ( bvsle #x00000000 y ) ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

