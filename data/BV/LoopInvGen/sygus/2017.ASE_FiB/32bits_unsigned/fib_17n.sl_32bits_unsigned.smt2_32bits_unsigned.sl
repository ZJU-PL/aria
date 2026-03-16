(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 32))(n (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))(k (_ BitVec 32))))

(define-fun pre_fun ((turn (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (k (_ BitVec 32))) Bool
       ( and ( = k #x00000001 ) ( = i #x00000001 ) ( = j #x00000000 ) ( = turn #x00000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 32)) (n! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k! (_ BitVec 32)) (turn (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (k (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( = turn #x00000001 ) ( bvult j i ) ( = k! ( bvsub ( bvadd k i ) j ) ) ( = i! i ) ( = j! ( bvadd j #x00000001 ) ) ( = n! n ) ( = turn! turn ) ) ) ) ( let ( ( a!2 ( or ( and ( = turn #x00000000 ) ( bvult i n ) ( = k! k ) ( = i! i ) ( = j! #x00000000 ) ( = n! n ) ( = turn! #x00000001 ) ) ( and ( = turn #x00000000 ) ( bvuge i n ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! #x00000003 ) ) a!1 ( and ( = turn #x00000001 ) ( bvuge j i ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000002 ) ( = k! k ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = n! n ) ( = turn! #x00000000 ) ) ( and ( bvuge turn #x00000003 ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! turn ) ) ( and ( bvult turn #x00000000 ) ( = k! k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = turn! turn ) ) ) ) ) ( and a!2 ) ) ))
(define-fun post_fun ((turn (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (k (_ BitVec 32))) Bool
       ( or ( not ( = turn #x00000003 ) ) ( bvule n k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

