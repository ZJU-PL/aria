; Petrobras problem
; ==========================================================
; http://ipc.icaps-conference.org/
; http://icaps12.poli.usp.br/icaps12/ickeps/petrobrasdomain
;

(define (domain petrobras)
    (:requirements :fluents :typing :universal-preconditions :disjunctive-preconditions :negative-preconditions :conditional-effects)
    
    (:types
     location locatable - object
     ship cargo - locatable
     port waiting_area platform - location 
    ) 

    (:predicates
        (at_ ?sh - locatable ?where - location)
        (can_refuel ?l - location)
        (docked ?sh - ship ?where - location)
        (loaded ?c - cargo ?sh - ship)
    )

    (:functions

     (current_fuel ?sh - ship) - number
     (current_load ?sh - ship) - number
     (fuel_capacity ?sh - ship) - number
     (load_capacity ?sh - ship) - number

     (current_docking_capacity ?p - location) - number
     (total_docking_capacity ?p - location) - number
     (distance ?from - location ?to - location) - number
     (weight ?c - cargo) - number
     (total_fuel_used) - number

    )

    (:action sail
            :parameters (?sh - ship ?from - location ?to - location)
            :precondition (and (at_ ?sh ?from)                           
                               (not (docked ?sh ?from))
                               (imply (= (current_load ?sh) 0)
                                   (>= (current_fuel ?sh) (/ (distance ?from ?to) 5)))
                               (imply (not (= (current_load ?sh) 0))
                                   (>= (current_fuel ?sh) (/ (distance ?from ?to) 3)))
)
                                
            :effect (and (at_ ?sh ?to)
                         (not (at_ ?sh ?from))
                         (when (= (current_load ?sh) 0) (and
                             (decrease (current_fuel ?sh) (/ (distance ?from ?to) 5))
                             (increase (total_fuel_used) (/ (distance ?from ?to) 5))))
                         (when (not (= (current_load ?sh) 0)) (and
                             (decrease (current_fuel ?sh) (/ (distance ?from ?to) 3))
                             (increase (total_fuel_used) (/ (distance ?from ?to) 3)))))
    )

    (:action dock
            :parameters (?sh - ship ?p - location)
            :precondition (and (> (current_docking_capacity ?p) 0)
                               (not (docked ?sh ?p))
                               (at_ ?sh ?p))
            :effect (and 
                        (decrease (current_docking_capacity ?p) 1)
                        (docked ?sh ?p))
    )

    (:action undock
            :parameters (?sh - ship ?p - location)
            :precondition (docked ?sh ?p)
            :effect (and 
                        (increase (current_docking_capacity ?p) 1)
                        (not (docked ?sh ?p)))
    )

    (:action load
            :parameters (?p - port ?sh - ship ?c - cargo)
            :precondition (and
			   (docked ?sh ?p)
                           (>= (load_capacity ?sh) (+ (current_load ?sh) (weight ?c)))
                          )

            :effect (and 
                        (not (at_ ?c ?p))
                        (loaded ?c ?sh)
                        (increase (current_load ?sh) (weight ?c)))
    )

    (:action unload
            :parameters (?c - cargo ?p - platform ?sh - ship)
            :precondition (and
                            (docked ?sh ?p)
                            (loaded ?c ?sh)
                          )
            :effect (and
                        (decrease (current_load ?sh) (weight ?c))
                        (not (loaded ?c ?sh))
                        (at_ ?c ?p))
    )

    (:action refuel_at_port
            :parameters (?sh - ship ?l - port)
            :precondition (and
                              (docked  ?sh ?l)
                              (can_refuel ?l)
                              (<= (+ (current_fuel ?sh) 200) (fuel_capacity ?sh))
                              )
            :effect (and 
                        (when (<= (+ (current_fuel ?sh) 200) (fuel_capacity ?sh))
                            (increase (current_fuel ?sh) 200))
                        (when (> (+ (current_fuel ?sh) 200) (fuel_capacity ?sh))
                            (assign (current_fuel ?sh) (fuel_capacity ?sh)))
	            )
	    )
    
    (:action refuel_at_platform
            :parameters (?sh - ship ?l - platform)
            :precondition (and
                              (docked  ?sh ?l)
                              (can_refuel ?l)
                              (<= (+ (current_fuel ?sh) 100) (fuel_capacity ?sh))
                              )
            :effect (and
                        (when (<= (+ (current_fuel ?sh) 100) (fuel_capacity ?sh))
                             (increase (current_fuel ?sh) 100))
                        (when (> (+ (current_fuel ?sh) 100) (fuel_capacity ?sh))
                            (assign (current_fuel ?sh) (fuel_capacity ?sh)))
                    )
	    )
)

