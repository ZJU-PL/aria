(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 64))(n (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))(k (_ BitVec 64))))

(define-fun pre_fun ((turn (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (k (_ BitVec 64))) Bool
       ( and ( = k #x0000000000000001 ) ( = i #x0000000000000001 ) ( = j #x0000000000000000 ) ( = turn #x0000000000000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 64)) (n! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k! (_ BitVec 64)) (turn (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (k (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( = turn #x0000000000000001 ) ( bvslt j i ) ( = k! ( bvsub ( bvadd k i ) j ) ) ( = i! i ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = n! n ) ( = turn! turn ) ) ) ) ( let ( ( a!2 ( or ( and ( = turn #x0000000000000000 ) ( bvslt i n ) ( = k! k ) ( = i! i ) ( = j! #x0000000000000000 ) ( = n! n ) ( = turn! #x0000000000000001 ) ) ( and ( = turn #x0000000000000000 ) ( bvsge i n ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! #x0000000000000003 ) ) a!1 ( and ( = turn #x0000000000000001 ) ( bvsge j i ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000002 ) ( = k! k ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! j ) ( = n! n ) ( = turn! #x0000000000000000 ) ) ( and ( bvsge turn #x0000000000000003 ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! turn ) ) ( and ( bvslt turn #x0000000000000000 ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! turn ) ) ) ) ) ( and a!2 ) ) ))
(define-fun post_fun ((turn (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (k (_ BitVec 64))) Bool
       ( or ( not ( = turn #x0000000000000003 ) ) ( bvsle n k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

