(set-logic QF_BV)
(declare-const x (_ BitVec 8))

; Constraints
(assert (bvuge (bvadd x #x05) #x0A))       ; x + 5 ≥ 10
(assert (= (bvand x #x0F) #x08))           ; x & 0x0F = 0x08
(assert (bvult x #x64))                     ; x < 100

; Optimization objective
(minimize x)

(check-sat)
(get-model)