;; Created by Drew McDermott for the IPC-1

;; This is the mystery domain with one extra action:
;;     the ability to squirt a unit of fuel from any node to a neighboring node
;;     provided the originating node has at least two units. 
(define (domain mystery-prime-typed)
;   (:requirements :typing :fluents)
   (:types food emotion - object
           pleasure pain - emotion)
   (:predicates
       (eats ?n1 ?n2 - food)
       (craves ?v - emotion ?n - food)
       (fears ?c - pain ?v - pleasure)
;       (locale ?n - food ?a - province)
;      (harmony ?v - emotion ?s - planet)
;       (attacks ?i ?j - province)
;       (orbits ?i ?j - planet)
   )

   (:functions
       (harmony ?v - emotion)
       (locale ?n - food)
   )

   (:action overcome
       :parameters (?c - pain ?v - pleasure ?n - food)
       :precondition (and (craves ?c ?n)
                          (craves ?v ?n)
                          (>= (harmony ?v) 1)
                     )
       :effect (and (not (craves ?c ?n))
                    (fears ?c ?v)
                    (decrease (harmony ?v) 1)
               )
   )

   (:action feast
       :parameters (?v - pleasure ?n1 ?n2 - food)
       :precondition (and (craves ?v ?n1)
                          (eats ?n1 ?n2)
                          (>= (locale ?n1) 1)
                     )
       :effect (and (not (craves ?v ?n1))
                    (craves ?v ?n2)
                    (decrease (locale ?n1) 1)
                    )
   )

   (:action succumb
       :parameters (?c - pain ?v - pleasure ?n - food)
       :precondition (and (fears ?c ?v)
                          (craves ?v ?n)
                     )
       :effect (and (not (fears ?c ?v))
                    (craves ?c ?n)
                    (increase (harmony ?v) 1)
                    ))
   (:action drink
      :parameters (?n1 ?n2 - food)
      :precondition (and (>= (locale ?n1) 1))
      :effect (and (decrease (locale ?n1) 1)
                   (increase (locale ?n2) 1))
   )
)
