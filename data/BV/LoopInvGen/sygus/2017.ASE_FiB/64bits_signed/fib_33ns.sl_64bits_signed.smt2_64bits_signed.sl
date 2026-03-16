(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 64))(k (_ BitVec 64))(c (_ BitVec 64))(z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((turn (_ BitVec 64)) (k (_ BitVec 64)) (c (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = z k ) ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ( = turn #x0000000000000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 64)) (k! (_ BitVec 64)) (c! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (turn (_ BitVec 64)) (k (_ BitVec 64)) (c (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( = turn #x0000000000000001 ) ( = z ( bvsub ( bvadd k y ) c ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = z! z ) ( = c! ( bvadd c #x0000000000000001 ) ) ( = k! k ) ( = turn! #x0000000000000001 ) ) ) ( a!2 ( not ( = z ( bvsub ( bvadd k y ) c ) ) ) ) ( a!3 ( and ( = turn #x0000000000000001 ) ( = z ( bvsub ( bvadd k y ) c ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = z! z ) ( = c! ( bvadd c #x0000000000000001 ) ) ( = k! k ) ( = turn! #x0000000000000002 ) ) ) ) ( let ( ( a!4 ( or ( and ( = turn #x0000000000000000 ) ( = x! x ) ( = y! y ) ( = z! z ) ( = c! #x0000000000000000 ) ( = k! k ) ( = turn! #x0000000000000001 ) ) ( and ( = turn #x0000000000000000 ) ( = x! x ) ( = y! y ) ( = z! z ) ( = c! #x0000000000000000 ) ( = k! k ) ( = turn! #x0000000000000002 ) ) a!1 ( and ( = turn #x0000000000000001 ) a!2 ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ( = z! z ) ( = c! ( bvadd c #x0000000000000001 ) ) ( = k! k ) ( = turn! #x0000000000000001 ) ) a!3 ( and ( = turn #x0000000000000001 ) a!2 ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ( = z! z ) ( = c! ( bvadd c #x0000000000000001 ) ) ( = k! k ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000002 ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ( = z! z ) ( = c! c ) ( = k! k ) ( = turn! #x0000000000000002 ) ) ( and ( = turn #x0000000000000002 ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ( = z! z ) ( = c! c ) ( = k! k ) ( = turn! #x0000000000000003 ) ) ( and ( or ( bvsgt turn #x0000000000000002 ) ( bvslt turn #x0000000000000000 ) ) ( = x! x ) ( = y! y ) ( = z! ( bvadd x y ) ) ( = c! c ) ( = k! k ) ( = turn! #x0000000000000000 ) ) ) ) ) ( and a!4 ) ) ))
(define-fun post_fun ((turn (_ BitVec 64)) (k (_ BitVec 64)) (c (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x y ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

