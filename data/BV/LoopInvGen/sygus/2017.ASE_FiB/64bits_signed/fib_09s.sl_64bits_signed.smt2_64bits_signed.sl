(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 64))(j (_ BitVec 64))(n (_ BitVec 64))(k (_ BitVec 64))(t (_ BitVec 64))(pvlen (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((turn (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64)) (k (_ BitVec 64)) (t (_ BitVec 64)) (pvlen (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = k #x0000000000000000 ) ( = i #x0000000000000000 ) ( = turn #x0000000000000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 64)) (j! (_ BitVec 64)) (n! (_ BitVec 64)) (k! (_ BitVec 64)) (t! (_ BitVec 64)) (pvlen! (_ BitVec 64)) (i! (_ BitVec 64)) (turn (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64)) (k (_ BitVec 64)) (t (_ BitVec 64)) (pvlen (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = turn #x0000000000000000 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000000 ) ) ( and ( = turn #x0000000000000000 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000001 ) ) ( and ( = turn #x0000000000000001 ) ( bvsgt i pvlen ) ( = i! #x0000000000000000 ) ( = pvlen! i ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000001 ) ( bvsle i pvlen ) ( = i! #x0000000000000000 ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000002 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = pvlen! pvlen ) ( = t! i ) ( = k! ( bvadd k #x0000000000000001 ) ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000002 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = pvlen! pvlen ) ( = t! i ) ( = k! ( bvadd k #x0000000000000001 ) ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000003 ) ) ( and ( = turn #x0000000000000003 ) ( = i! i ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000003 ) ) ( and ( = turn #x0000000000000003 ) ( = i! i ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x0000000000000004 ) ) ( and ( = turn #x0000000000000004 ) ( = i! i ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! i ) ( = j! #x0000000000000000 ) ( = turn! #x0000000000000005 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((turn (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64)) (k (_ BitVec 64)) (t (_ BitVec 64)) (pvlen (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( bvsle #x0000000000000000 k ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

