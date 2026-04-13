;; Simplified Block Grouping Problem with 3 Blocks
;; Based on work by Enrico Scala and Miquel Ramirez

(define (problem simplified-block-grouping-3)
  (:domain mt-block-grouping)
  
  ;; Define only 3 blocks
  (:objects
    ;; Group 1 blocks (should end up together)
    b1 b2 - block
    
    ;; Group 2 block (should be separate from Group 1)
    b3 - block
  )

  ;; Initial block positions on a 5x5 grid
  (:init
    ;; Group 1 initial positions
    (= (x b1) 1) (= (y b1) 3)
    (= (x b2) 2) (= (y b2) 4)
    
    ;; Group 2 initial position
    (= (x b3) 4) (= (y b3) 3)
    
    ;; Grid boundaries (5x5 grid from 1,1 to 5,5)
    (= (max_x) 5)
    (= (min_x) 1)
    (= (max_y) 5)
    (= (min_y) 1)
  )

  ;; Goal: Group blocks by color
  ;; Group 1 (b1, b2) should be together
  ;; Group 2 (b3) should be in a different location
  (:goal (and 
    ;; Group 1 blocks must be together
    (= (x b1) (x b2)) (= (y b1) (y b2))
    
    ;; Group 1 and Group 2 must be separated
    (or (not (= (x b1) (x b3))) (not (= (y b1) (y b3))))
  ))
)