(set-logic BV)

(synth-inv inv_fun ((v (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))(z (_ BitVec 32))))

(define-fun pre_fun ((v (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( and ( bvsgt x y ) ( bvsgt y z ) ( = v #x00000000 ) ))
(define-fun trans_fun ((v! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (v (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000003 ) ) ( = z! ( bvadd z #x00000002 ) ) ( = v! ( ite ( bvslt x y ) ( bvadd v #x00000001 ) v ) ) ))
(define-fun post_fun ((v (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( bvsle ( bvadd z ( bvmul #xffffffff x ) ) #x00011b53 ) ) ) ) ( not ( and a!1 ( not ( bvsle v #x00000000 ) ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

