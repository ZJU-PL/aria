(define (problem minimal-petrobras)
 (:domain petrobras)

 (:objects
  ship1 - ship
  port1 - port
  platform1 - platform
  cargo1 - cargo
 )

 (:init
  (= (total_fuel_used) 0)

  ;; Ship initial state
  (at_ ship1 port1)
  (docked ship1 port1)
  (= (current_fuel ship1) 200)
  (= (current_load ship1) 0)
  (= (fuel_capacity ship1) 200)
  (= (load_capacity ship1) 50)

  ;; Location properties
  (= (current_docking_capacity port1) 1)
  (= (total_docking_capacity port1) 1)
  (= (current_docking_capacity platform1) 1)
  (= (total_docking_capacity platform1) 1)
  
  ;; Refueling capabilities
  (can_refuel port1)

  ;; Cargo
  (at_ cargo1 port1)
  (= (weight cargo1) 20)

  ;; Distances
  (= (distance port1 port1) 999)
  (= (distance platform1 platform1) 999)
  (= (distance port1 platform1) 100)
  (= (distance platform1 port1) 100)
 )

 (:goal
  (and
   (at_ cargo1 platform1)
  )
 )
)