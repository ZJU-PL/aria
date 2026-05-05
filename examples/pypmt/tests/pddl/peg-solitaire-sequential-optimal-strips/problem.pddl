(define (problem minimal-pegsolitaire)
  (:domain pegsolitaire-sequential)
  
  ;; Define the simplest layout: 3 locations in a row
  (:objects
    loc1 loc2 loc3 - location
  )
  
  (:init
    ;; Define the line relationship
    (IN-LINE loc1 loc2 loc3)
    
    ;; Initial state: first and second locations occupied, third is free
    (occupied loc1)
    (occupied loc2)
    (free loc3)
    
    ;; Move sequence control
    (move-ended)
    
    ;; Initialize total cost
    (= (total-cost) 0)
  )
  
  (:goal
    (and
      ;; Goal: first and second locations free, third occupied
      (free loc1)
      (free loc2)
      (occupied loc3)
      ;; Ensure move is ended for goal state
      (move-ended)
    )
  )
  
  ;; Specify metric to minimize
  (:metric minimize (total-cost))
)