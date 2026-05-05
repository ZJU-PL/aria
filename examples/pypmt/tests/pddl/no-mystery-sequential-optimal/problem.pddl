(define (problem minimal-transport)
  (:domain transport-strips)
  
  (:objects
    loc1 loc2 - location
    package1 - package
    truck1 - truck
    fuel0 fuel1 fuel2 - fuellevel
  )
  
  (:init
    ;; Connectivity between locations
    (connected loc1 loc2)
    (connected loc2 loc1)
    
    ;; Initial positions
    (at truck1 loc1)
    (at package1 loc1)
    
    ;; Initial fuel level
    (fuel truck1 fuel2)
    
    ;; Fuel costs
    (fuelcost fuel1 loc1 loc2)
    (fuelcost fuel1 loc2 loc1)
    
    ;; Fuel arithmetic
    (sum fuel0 fuel1 fuel1)
    (sum fuel1 fuel1 fuel2)
    
    ;; Initial cost
    (= (total-cost) 0)
  )
  
  (:goal
    (and
      (at package1 loc2)
    )
  )
  
  ;; Specify metric to minimize
  (:metric minimize (total-cost))
)