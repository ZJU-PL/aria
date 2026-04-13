(define (problem minimal-scanalyzer)
  (:domain scanalyzer3d)
  (:objects
    car-a - car
    car-b - car
    seg-1 - segment
    seg-2 - segment
  )
  (:init
    (= (total-cost) 0)
    
    ;; Define the single rotation cycle
    (CYCLE-2 seg-1 seg-2)
    
    ;; Define the cycle with analysis
    (CYCLE-2-WITH-ANALYSIS seg-1 seg-2)
    
    ;; Initial positions of cars
    (on car-a seg-1)
    (on car-b seg-2)
  )
  (:goal (and
    ;; Both cars must be analyzed
    (analyzed car-a)
    (analyzed car-b)
    
    ;; Cars must return to their original positions
    (on car-a seg-1)
    (on car-b seg-2)
  ))
  (:metric minimize (total-cost))
)