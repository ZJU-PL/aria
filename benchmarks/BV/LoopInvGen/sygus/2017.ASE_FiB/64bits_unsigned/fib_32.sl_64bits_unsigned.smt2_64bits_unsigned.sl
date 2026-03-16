(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))(b (_ BitVec 64))(k (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (b (_ BitVec 64)) (k (_ BitVec 64))) Bool
       ( and ( bvugt k #x0000000000000000 ) ( = i j ) ( = n #x0000000000000000 ) ( or ( = b #x0000000000000001 ) ( = b #x0000000000000000 ) ) ))
(define-fun trans_fun ((n! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (b! (_ BitVec 64)) (k! (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (b (_ BitVec 64)) (k (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvuge n ( bvmul #x0000000000000002 k ) ) ( = k! k ) ( = b! b ) ( = i! i ) ( = j! j ) ( = n! n ) ) ( and ( bvult n ( bvmul #x0000000000000002 k ) ) ( = b #x0000000000000001 ) ( = k! k ) ( = b! #x0000000000000000 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! j ) ( = n! ( bvadd n #x0000000000000001 ) ) ) ( and ( bvult n ( bvmul #x0000000000000002 k ) ) ( not ( = b #x0000000000000001 ) ) ( = k! k ) ( = b! #x0000000000000001 ) ( = i! i ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = n! ( bvadd n #x0000000000000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (b (_ BitVec 64)) (k (_ BitVec 64))) Bool
       ( or ( not ( bvule ( bvmul #x0000000000000002 k ) n ) ) ( = i j ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

