(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 32))(j (_ BitVec 32))(n (_ BitVec 32))(k (_ BitVec 32))(t (_ BitVec 32))(pvlen (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((turn (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32)) (k (_ BitVec 32)) (t (_ BitVec 32)) (pvlen (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = k #x00000000 ) ( = i #x00000000 ) ( = turn #x00000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 32)) (j! (_ BitVec 32)) (n! (_ BitVec 32)) (k! (_ BitVec 32)) (t! (_ BitVec 32)) (pvlen! (_ BitVec 32)) (i! (_ BitVec 32)) (turn (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32)) (k (_ BitVec 32)) (t (_ BitVec 32)) (pvlen (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = turn #x00000000 ) ( = i! ( bvadd i #x00000001 ) ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x00000000 ) ) ( and ( = turn #x00000000 ) ( = i! ( bvadd i #x00000001 ) ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x00000001 ) ) ( and ( = turn #x00000001 ) ( bvsgt i pvlen ) ( = i! #x00000000 ) ( = pvlen! i ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000001 ) ( bvsle i pvlen ) ( = i! #x00000000 ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000002 ) ( = i! ( bvadd i #x00000001 ) ) ( = pvlen! pvlen ) ( = t! i ) ( = k! ( bvadd k #x00000001 ) ) ( = n! n ) ( = j! j ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000002 ) ( = i! ( bvadd i #x00000001 ) ) ( = pvlen! pvlen ) ( = t! i ) ( = k! ( bvadd k #x00000001 ) ) ( = n! n ) ( = j! j ) ( = turn! #x00000003 ) ) ( and ( = turn #x00000003 ) ( = i! i ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x00000003 ) ) ( and ( = turn #x00000003 ) ( = i! i ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! n ) ( = j! j ) ( = turn! #x00000004 ) ) ( and ( = turn #x00000004 ) ( = i! i ) ( = pvlen! pvlen ) ( = t! t ) ( = k! k ) ( = n! i ) ( = j! #x00000000 ) ( = turn! #x00000005 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((turn (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32)) (k (_ BitVec 32)) (t (_ BitVec 32)) (pvlen (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( bvsle #x00000000 k ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

