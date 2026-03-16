(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((turn (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ( = i #x0000000000000000 ) ( = j #x0000000000000000 ) ( = turn #x0000000000000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (turn (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = turn #x0000000000000000 ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! j ) ( = turn! #x0000000000000001 ) ) ( and ( = turn #x0000000000000000 ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! j ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000001 ) ( = x y ) ( = x! x ) ( = y! y ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! j ) ( = turn! #x0000000000000001 ) ) ( and ( = turn #x0000000000000001 ) ( = x y ) ( = x! x ) ( = y! y ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! j ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000001 ) ( not ( = x y ) ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = turn! #x0000000000000001 ) ) ( and ( = turn #x0000000000000001 ) ( not ( = x y ) ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000002 ) ( bvsge i j ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = i! i ) ( = j! j ) ( = turn! #x0000000000000000 ) ) ( and ( = turn #x0000000000000002 ) ( bvslt i j ) ( = x! x ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = i! i ) ( = j! j ) ( = turn! #x0000000000000000 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((turn (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( bvsle j i ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

