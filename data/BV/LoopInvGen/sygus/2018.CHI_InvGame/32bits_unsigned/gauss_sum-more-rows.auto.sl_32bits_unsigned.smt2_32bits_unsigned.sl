(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 32))(sum (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((i (_ BitVec 32)) (sum (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( bvule #x00000001 n ) ( bvule n #x000003e8 ) ( = sum #x00000000 ) ( = i #x00000001 ) ))
(define-fun trans_fun ((i! (_ BitVec 32)) (sum! (_ BitVec 32)) (n! (_ BitVec 32)) (i (_ BitVec 32)) (sum (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( bvule i n ) ( = i! ( bvadd i #x00000001 ) ) ( = sum! ( bvadd sum i ) ) ( = n! n ) ))
(define-fun post_fun ((i (_ BitVec 32)) (sum (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( bvule i n ) ( = ( bvmul #x00000002 sum ) ( bvmul n ( bvadd #x00000001 n ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

