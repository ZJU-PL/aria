; from  Hacker's delight
; Population Count (Counting Set Bits):

(set-logic QF_BV)
(declare-fun x () (_ BitVec 32))
(declare-fun step1 () (_ BitVec 32))
(declare-fun step2 () (_ BitVec 32))
(declare-fun result () (_ BitVec 32))

; Step 1: Count bits in parallel using magic numbers
(assert (= step1 (bvsub x (bvand (bvlshr x #x00000001) #x55555555))))

; Step 2: Sum 2-bit fields
(assert (= step2 (bvadd (bvand step1 #x33333333)
                       (bvand (bvlshr step1 #x00000002) #x33333333))))

; Final result: Sum 4-bit fields and return
(assert (= result (bvand (bvadd (bvand step2 #x0F0F0F0F)
                               (bvand (bvlshr step2 #x00000004) #x0F0F0F0F))
                        #x000000FF)))

; Verify for some specific cases
(assert (=> (= x #x00000000) (= result #x00000000)))
(assert (=> (= x #xFFFFFFFF) (= result #x00000020)))

; Optimize for minimum number of operations
(minimize (bvadd (ite (= step1 x) #x00000001 #x00000000)
                (ite (= step2 step1) #x00000001 #x00000000)
                (ite (= result step2) #x00000001 #x00000000)))
(check-sat)
;(get-model)