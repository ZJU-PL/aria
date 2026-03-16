(set-logic BV)

(synth-inv inv_fun ((flag (_ BitVec 64))(n (_ BitVec 64))(j (_ BitVec 64))(b (_ BitVec 64))))

(define-fun pre_fun ((flag (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (b (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( bvsgt n #x0000000000000000 ) ( = b #x0000000000000000 ) ))
(define-fun trans_fun ((flag! (_ BitVec 64)) (n! (_ BitVec 64)) (j! (_ BitVec 64)) (b! (_ BitVec 64)) (flag (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (b (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt b n ) ( = flag #x0000000000000001 ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = b! ( bvadd b #x0000000000000001 ) ) ( = n! n ) ( = flag! flag ) ) ( and ( bvslt b n ) ( not ( = flag #x0000000000000001 ) ) ( = j! j ) ( = b! ( bvadd b #x0000000000000001 ) ) ( = n! n ) ( = flag! flag ) ) ( and ( bvsge b n ) ( = j! j ) ( = b! b ) ( = n! n ) ( = flag! flag ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((flag (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (b (_ BitVec 64))) Bool
       ( or ( not ( bvsle n b ) ) ( = j n ) ( not ( = flag #x0000000000000001 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

