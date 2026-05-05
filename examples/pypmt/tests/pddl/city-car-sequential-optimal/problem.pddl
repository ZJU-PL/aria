(define (problem minimal-citycar)
  (:domain citycar)
  
  (:objects
    ;; Define minimal objects needed
    car1 - car
    j1 j2 - junction
    garage1 - garage
    road1 - road
  )
  
  (:init
    ;; Junction relationships - j1 and j2 can be connected both ways
    (same_line j1 j2)
    
    ;; Initial junction states - both junctions are clear initially
    (clear j1)
    (clear j2)
    
    ;; Garage setup - garage is at j1
    (at_garage garage1 j1)
    (starting car1 garage1)
    
    ;; No roads are in place initially
    (not (in_place road1))
    
    ;; Initial cost
    (= (total-cost) 0)
  )
  
  (:goal
    (and
      ;; Car should arrive at junction j2
      (arrived car1 j2)
    )
  )
  
  ;; Specify metric to minimize
  (:metric minimize (total-cost))
)