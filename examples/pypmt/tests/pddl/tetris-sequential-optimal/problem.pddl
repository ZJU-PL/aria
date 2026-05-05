(define (problem minimal-tetris)
(:domain tetris)
(:objects  
p0-0 p0-1 
p1-0 p1-1 - position
square0 - one_square
)
(:init
;; Define connectivity (2x2 grid)
;; Horizontal connections
(connected p0-0 p0-1)
(connected p0-1 p0-0)
(connected p1-0 p1-1)
(connected p1-1 p1-0)

;; Vertical connections
(connected p0-0 p1-0)
(connected p1-0 p0-0)
(connected p0-1 p1-1)
(connected p1-1 p0-1)

;; Initial state - just one square piece at top-left
(at_square square0 p0-0)

;; Clear positions
(clear p0-1)
(clear p1-0)
(clear p1-1)

;; Total cost starts at 0
(= (total-cost) 0)
)
(:goal
(and
(at_square square0 p1-1)
)
)
(:metric minimize (total-cost))
)
;; DESCRIPTION OF THE INITIAL STATE
;; 0   S*  
;; 1   **  
;;
;; S = square piece
;; * = empty space