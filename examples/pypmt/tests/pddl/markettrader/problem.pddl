(define (problem minimal-trader)
(:domain Trader)
(:objects
  Market1 Market2 - market
  camel1 - camel
  Apple Banana - goods
)
(:init
  ;; Prices and availability at Market1
  (= (price Apple Market1) 5)
  (= (on-sale Apple Market1) 10)
  (= (price Banana Market1) 20)
  (= (on-sale Banana Market1) 0)
  
  ;; Prices and availability at Market2
  (= (price Apple Market2) 15)
  (= (on-sale Apple Market2) 0)
  (= (price Banana Market2) 10)
  (= (on-sale Banana Market2) 5)
  
  ;; Initial state
  (= (bought Apple) 0)
  (= (bought Banana) 0)
  (= (drive-cost Market1 Market2) 2)
  (= (drive-cost Market2 Market1) 2)
  (can-drive Market1 Market2)
  (can-drive Market2 Market1)
  (at camel1 Market1)
  (= (cash) 20)
  (= (capacity) 2)
)
(:goal (and
  (>= (cash) 40)
))
)