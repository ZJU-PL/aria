(define (problem minimal-parking)
  (:domain parking)
  
  ;; Define cars and curbs - need at least 3 curbs for a solvable problem
  (:objects
    car1 car2 - car
    curb1 curb2 curb3 - curb
  )
  
  (:init
    ;; Initial positions of cars
    (at-curb car1)
    (at-curb-num car1 curb1)
    (car-clear car1)
    
    (at-curb car2)
    (at-curb-num car2 curb2)
    (car-clear car2)
    
    ;; curb3 is clear initially
    (curb-clear curb3)
    
    ;; Curbs with cars are not clear
    (not (curb-clear curb1))
    (not (curb-clear curb2))
    
    ;; Initial cost
    (= (total-cost) 0)
  )
  
  (:goal
    (and
      ;; Goal: cars are in swapped positions
      (at-curb-num car1 curb2)
      (at-curb-num car2 curb1)
    )
  )
  
  ;; Specify metric to minimize
  (:metric minimize (total-cost))
)