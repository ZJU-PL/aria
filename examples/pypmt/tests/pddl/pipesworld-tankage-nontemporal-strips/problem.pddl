(define (problem minimal-pipesworld)
  (:domain pipesworld_strips)
  (:objects
    ;; Two batch-atoms
    B0 B1 - batch-atom
    
    ;; Two areas
    A1 A2 - area
    
    ;; One pipe
    S12 - pipe
    
    ;; Tank slots - one for each product in each area
    TA1-lco TA2-lco - tank-slot
  )
  
  (:init
    ;; Pipeline is in normal state
    (normal S12)
    
    ;; Network topology - one direction only, as in original
    (connect A1 A2 S12)
    
    ;; Define pipe as unitary
    (unitary S12)
    
    ;; Product interfaces
    (may-interface lco lco)
    
    ;; Tank slot locations
    (tank-slot-product-location TA1-lco lco A1)
    (tank-slot-product-location TA2-lco lco A2)
    
    ;; Batch-atoms products - both are lco
    (is-product B0 lco)
    (is-product B1 lco)
    
    ;; Initial batch locations
    (on B0 A1)
    (occupied TA1-lco)
    
    ;; Available tank slots
    (not-occupied TA2-lco)
    
    ;; Batch in pipe
    (first B1 S12)
    (last B1 S12)
  )
  
  (:goal 
    (and
      (on B1 A2)
      (normal S12)
    )
  )
)