(define (problem minimal-pipesworld)
  (:domain pipesworld_strips)
  (:objects
    ;; Just 2 batches
    B0 B4 - batch-atom
    
    ;; 2 areas
    A1 A2 - area
    
    ;; 1 pipe
    S12 - pipe
    
    ;; Only lco tank slots
    TA1-1-lco - tank-slot
    TA2-1-lco - tank-slot
  )
  
  (:init
    ;; Pipeline is in normal state
    (normal S12)
    
    ;; Interface restrictions
    (may-interface lco lco)
    
    ;; Network topology
    (connect A1 A2 S12)
    
    ;; Tank slot locations
    (tank-slot-product-location TA1-1-lco lco A1)
    (tank-slot-product-location TA2-1-lco lco A2)
    
    ;; Batch-atoms products
    (is-product B0 lco)
    (is-product B4 lco)
    
    ;; Batch-atoms initially located in areas
    (on B0 A1)
    (occupied TA1-1-lco)
    
    (not-occupied TA2-1-lco)
    
    ;; Batch-atoms initially located in pipes
    (first B4 S12)
    (last B4 S12)
    
    ;; Unitary pipeline segments
    (unitary S12)
  )
  
  (:goal (and
    (on B4 A2)
    (normal S12)
  ))
)