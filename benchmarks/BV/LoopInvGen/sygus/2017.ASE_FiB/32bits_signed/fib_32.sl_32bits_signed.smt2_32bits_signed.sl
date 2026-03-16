(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))(b (_ BitVec 32))(k (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (b (_ BitVec 32)) (k (_ BitVec 32))) Bool
       ( and ( bvsgt k #x00000000 ) ( = i j ) ( = n #x00000000 ) ( or ( = b #x00000001 ) ( = b #x00000000 ) ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (b! (_ BitVec 32)) (k! (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (b (_ BitVec 32)) (k (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvsge n ( bvmul #x00000002 k ) ) ( = k! k ) ( = b! b ) ( = i! i ) ( = j! j ) ( = n! n ) ) ( and ( bvslt n ( bvmul #x00000002 k ) ) ( = b #x00000001 ) ( = k! k ) ( = b! #x00000000 ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = n! ( bvadd n #x00000001 ) ) ) ( and ( bvslt n ( bvmul #x00000002 k ) ) ( not ( = b #x00000001 ) ) ( = k! k ) ( = b! #x00000001 ) ( = i! i ) ( = j! ( bvadd j #x00000001 ) ) ( = n! ( bvadd n #x00000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (b (_ BitVec 32)) (k (_ BitVec 32))) Bool
       ( or ( not ( bvsle ( bvmul #x00000002 k ) n ) ) ( = i j ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

