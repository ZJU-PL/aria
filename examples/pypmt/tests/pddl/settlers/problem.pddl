(define (problem simple-civ)
  (:domain civ)
  
  (:objects
    village port - place
    cart1 - vehicle
  )
  
  (:init
    ;; Connectivity
    (connected-by-land village port)
    (by-coast port)
    
    ;; Geography
    (woodland village)
    (mountain port)
    
    ;; Vehicle info
    (potential cart1)
    
    ;; Initial resources
    (= (available timber village) 3)
    (= (available wood village) 0)
    (= (available coal village) 0)
    (= (available stone village) 0)
    (= (available iron village) 0)
    (= (available ore village) 0)
    
    (= (available timber port) 0)
    (= (available wood port) 0)
    (= (available coal port) 0)
    (= (available stone port) 0)
    (= (available iron port) 0)
    (= (available ore port) 0)
    
    ;; Initial state values
    (= (carts-at village) 0)
    (= (carts-at port) 0)
    (= (labour) 0)
    (= (resource-use) 0)
    (= (pollution) 0)
    (= (housing village) 0)
    (= (housing port) 0)
  )
  
  (:goal
    (and
      ;; Build basic infrastructure
      (has-cabin village)
      (has-quarry port)
      
      ;; Produce some resources
      (> (available timber village) 0)
      (> (available stone port) 0)
    )
  )
  
  ;; Simple metric to minimize labor costs
  (:metric minimize (labour))
)