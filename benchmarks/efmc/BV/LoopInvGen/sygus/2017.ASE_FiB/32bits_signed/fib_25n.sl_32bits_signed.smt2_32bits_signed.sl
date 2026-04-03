(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((turn (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ( = i #x00000000 ) ( = j #x00000000 ) ( = turn #x00000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (turn (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = turn #x00000000 ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! j ) ( = turn! #x00000001 ) ) ( and ( = turn #x00000000 ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! j ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000001 ) ( = x y ) ( = x! x ) ( = y! y ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = turn! #x00000001 ) ) ( and ( = turn #x00000001 ) ( = x y ) ( = x! x ) ( = y! y ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000001 ) ( not ( = x y ) ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! ( bvadd j #x00000001 ) ) ( = turn! #x00000001 ) ) ( and ( = turn #x00000001 ) ( not ( = x y ) ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! ( bvadd j #x00000001 ) ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000002 ) ( bvsge i j ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = i! i ) ( = j! j ) ( = turn! #x00000000 ) ) ( and ( = turn #x00000002 ) ( bvslt i j ) ( = x! x ) ( = y! ( bvadd y #x00000001 ) ) ( = i! i ) ( = j! j ) ( = turn! #x00000000 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((turn (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( bvsle j i ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

