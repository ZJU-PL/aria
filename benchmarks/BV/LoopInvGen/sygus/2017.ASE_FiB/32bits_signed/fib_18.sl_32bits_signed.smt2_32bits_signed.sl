(set-logic BV)

(synth-inv inv_fun ((flag (_ BitVec 32))(n (_ BitVec 32))(j (_ BitVec 32))(b (_ BitVec 32))))

(define-fun pre_fun ((flag (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (b (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( bvsgt n #x00000000 ) ( = b #x00000000 ) ))
(define-fun trans_fun ((flag! (_ BitVec 32)) (n! (_ BitVec 32)) (j! (_ BitVec 32)) (b! (_ BitVec 32)) (flag (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (b (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt b n ) ( = flag #x00000001 ) ( = j! ( bvadd j #x00000001 ) ) ( = b! ( bvadd b #x00000001 ) ) ( = n! n ) ( = flag! flag ) ) ( and ( bvslt b n ) ( not ( = flag #x00000001 ) ) ( = j! j ) ( = b! ( bvadd b #x00000001 ) ) ( = n! n ) ( = flag! flag ) ) ( and ( bvsge b n ) ( = j! j ) ( = b! b ) ( = n! n ) ( = flag! flag ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((flag (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (b (_ BitVec 32))) Bool
       ( or ( not ( = flag #x00000001 ) ) ( not ( bvsle n b ) ) ( = j n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

