(set-logic BV)

(synth-inv inv_fun ((v (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))(z (_ BitVec 64))))

(define-fun pre_fun ((v (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( and ( bvsgt x y ) ( bvsgt y z ) ( = v #x0000000000000000 ) ))
(define-fun trans_fun ((v! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (v (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000003 ) ) ( = z! ( bvadd z #x0000000000000002 ) ) ( = v! ( ite ( bvslt x y ) ( bvadd v #x0000000000000001 ) v ) ) ))
(define-fun post_fun ((v (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( bvsle ( bvadd z ( bvmul #xffffffffffffffff x ) ) #x0000000000011b53 ) ) ) ) ( not ( and a!1 ( bvsle v #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

