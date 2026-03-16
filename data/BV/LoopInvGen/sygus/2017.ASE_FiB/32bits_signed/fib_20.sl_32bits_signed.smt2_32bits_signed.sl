(set-logic BV)

(synth-inv inv_fun ((m (_ BitVec 32))(n (_ BitVec 32))(i (_ BitVec 32))(j (_ BitVec 32))(k (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((m (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32)) (j (_ BitVec 32)) (k (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = ( bvadd x y ) k ) ( = m #x00000000 ) ( = j #x00000000 ) ))
(define-fun trans_fun ((m! (_ BitVec 32)) (n! (_ BitVec 32)) (i! (_ BitVec 32)) (j! (_ BitVec 32)) (k! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (m (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32)) (j (_ BitVec 32)) (k (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt j n ) ( = j i ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( = k! k ) ( = j! ( bvadd j #x00000001 ) ) ( = i! i ) ( = n! n ) ( = m! m ) ) ( and ( bvslt j n ) ( = j i ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( = k! k ) ( = j! ( bvadd j #x00000001 ) ) ( = i! i ) ( = n! n ) ( = m! j ) ) ( and ( bvslt j n ) ( not ( = j i ) ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = k! k ) ( = j! ( bvadd j #x00000001 ) ) ( = i! i ) ( = n! n ) ( = m! m ) ) ( and ( bvslt j n ) ( not ( = j i ) ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = k! k ) ( = j! ( bvadd j #x00000001 ) ) ( = i! i ) ( = n! n ) ( = m! j ) ) ( and ( bvsge j n ) ( = x! x ) ( = y! y ) ( = k! k ) ( = j! j ) ( = i! i ) ( = n! n ) ( = m! m ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((m (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32)) (j (_ BitVec 32)) (k (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle n j ) ) ( and ( = ( bvadd x y ) k ) ( or ( bvsle n #x00000000 ) ( bvsle #x00000000 m ) ) ( or ( bvsle n #x00000000 ) ( bvsle m n ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

