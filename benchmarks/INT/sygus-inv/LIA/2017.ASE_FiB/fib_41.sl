(set-logic LIA)

(synth-inv inv_fun ((n Int) (k Int) (i Int) (j Int)))

(define-fun pre_fun ((n Int) (k Int) (i Int) (j Int)) Bool
    (and (= j 0) (>= n 0) (= i 0) (or (= k 1) (>= k 0))))
(define-fun trans_fun ((n Int) (k Int) (i Int) (j Int) (n! Int) (k! Int) (i! Int) (j! Int)) Bool
    (or (and (> i n) (= n! n) (= k! k) (= i! i) (= j! j)) (and (<= i n) (= n! n) (= k! k) (= i! (+ i 1)) (= j! (+ j 1)))))
(define-fun post_fun ((n Int) (k Int) (i Int) (j Int)) Bool
    (=> (> i n) (> (+ k i j) (* 2 n))))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

; (define-fun inv_fun ((n Int) (k Int) (i Int) (j Int)) Bool (let ((_let_1 (* (- 1) i))) (and (or (>= (+ n _let_1) 0) (not (>= (+ (* 2 n) (* (- 1) k) _let_1 (* (- 1) j)) 0))) (>= (+ k _let_1 j) 0))))
