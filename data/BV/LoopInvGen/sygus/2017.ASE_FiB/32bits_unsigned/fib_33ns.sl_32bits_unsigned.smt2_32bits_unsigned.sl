(set-logic BV)

(synth-inv inv_fun ((turn (_ BitVec 32))(k (_ BitVec 32))(c (_ BitVec 32))(z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((turn (_ BitVec 32)) (k (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = z k ) ( = x #x00000000 ) ( = y #x00000000 ) ( = turn #x00000000 ) ))
(define-fun trans_fun ((turn! (_ BitVec 32)) (k! (_ BitVec 32)) (c! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (turn (_ BitVec 32)) (k (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( = turn #x00000001 ) ( = z ( bvsub ( bvadd k y ) c ) ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = z! z ) ( = c! ( bvadd c #x00000001 ) ) ( = k! k ) ( = turn! #x00000001 ) ) ) ( a!2 ( not ( = z ( bvsub ( bvadd k y ) c ) ) ) ) ( a!3 ( and ( = turn #x00000001 ) ( = z ( bvsub ( bvadd k y ) c ) ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = z! z ) ( = c! ( bvadd c #x00000001 ) ) ( = k! k ) ( = turn! #x00000002 ) ) ) ) ( let ( ( a!4 ( or ( and ( = turn #x00000000 ) ( = x! x ) ( = y! y ) ( = z! z ) ( = c! #x00000000 ) ( = k! k ) ( = turn! #x00000001 ) ) ( and ( = turn #x00000000 ) ( = x! x ) ( = y! y ) ( = z! z ) ( = c! #x00000000 ) ( = k! k ) ( = turn! #x00000002 ) ) a!1 ( and ( = turn #x00000001 ) a!2 ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( = z! z ) ( = c! ( bvadd c #x00000001 ) ) ( = k! k ) ( = turn! #x00000001 ) ) a!3 ( and ( = turn #x00000001 ) a!2 ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( = z! z ) ( = c! ( bvadd c #x00000001 ) ) ( = k! k ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000002 ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( = z! z ) ( = c! c ) ( = k! k ) ( = turn! #x00000002 ) ) ( and ( = turn #x00000002 ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( = z! z ) ( = c! c ) ( = k! k ) ( = turn! #x00000003 ) ) ( and ( or ( bvugt turn #x00000002 ) ( bvult turn #x00000000 ) ) ( = x! x ) ( = y! y ) ( = z! ( bvadd x y ) ) ( = c! c ) ( = k! k ) ( = turn! #x00000000 ) ) ) ) ) ( and a!4 ) ) ))
(define-fun post_fun ((turn (_ BitVec 32)) (k (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x y ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

