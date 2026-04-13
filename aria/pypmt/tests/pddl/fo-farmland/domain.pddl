;; Enrico Scala (enricos83@gmail.com) and Dongxu Li (dongxu.li@anu.edu.au,)
;; Reference Paper: Li, D., Scala, E., Haslum, P., & Bogomolov, S. (2018, July). 
;;                  Effect-abstraction based relaxation for linear numeric planning. 
;;                  In Proceedings of the 27th International 
;;                  Joint Conference on Artificial Intelligence (pp. 4787-4793).
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; This is an extended version of the Farmland domain; it does so introducing a new action (movebycar)
;; with which it is possible to move more workers per time from place to place, yet paying a higher cost

(define (domain farmland_ln)
    (:requirements :strips :fluents :adl)
    (:types farm -object
            
    )
    (:predicates (adj ?f1 ?f2 - farm) (dummy))
    (:functions
        (x ?b - farm)
        (cost)
        (num-of-cars)
    )

    ;; Move a person from a unit f1 to a unit f2
    (:action move-by-car
        :parameters (?f1 ?f2 - farm)
        :precondition (and (not (= ?f1 ?f2)) (>= (x ?f1) (* 4 (num-of-cars))) (adj ?f1 ?f2) )
        :effect (and  (decrease (x ?f1) (* 4 (num-of-cars))) 
                      (increase (x ?f2) (* 4 (num-of-cars)))
                      (increase (cost) (* 0.1  (* 4 (num-of-cars))))
                )
    )
    
    (:action move-slow
         :parameters (?f1 ?f2 - farm)
         :precondition (and (not (= ?f1 ?f2)) (>= (x ?f1) 1) (adj ?f1 ?f2))
         :effect (and(decrease (x ?f1) 1) (increase (x ?f2) 1))
    )

    (:action hire-car
        :parameters ()
	:precondition ( and (not (dummy)))
        :effect (and  (increase (num-of-cars) 1))
    )
)
