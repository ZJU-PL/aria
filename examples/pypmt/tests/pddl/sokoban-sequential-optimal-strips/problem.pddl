(define (problem mini-sokoban)
  (:domain sokoban-sequential)
  (:objects
    dir-down - direction
    dir-left - direction
    dir-right - direction
    dir-up - direction
    player-01 - player
    stone-01 - stone
    pos-1-1 - location
    pos-1-2 - location
    pos-1-3 - location
    pos-2-1 - location
    pos-2-2 - location
    pos-2-3 - location
    pos-3-1 - location
    pos-3-2 - location
    pos-3-3 - location
  )
  (:init
    ;; Set goal location
    (IS-GOAL pos-3-3)
    
    ;; Set non-goal locations
    (IS-NONGOAL pos-1-1)
    (IS-NONGOAL pos-1-2)
    (IS-NONGOAL pos-1-3)
    (IS-NONGOAL pos-2-1)
    (IS-NONGOAL pos-2-2)
    (IS-NONGOAL pos-2-3)
    (IS-NONGOAL pos-3-1)
    (IS-NONGOAL pos-3-2)
    
    ;; Define valid moves
    (MOVE-DIR pos-1-1 pos-1-2 dir-down)
    (MOVE-DIR pos-1-1 pos-2-1 dir-right)
    (MOVE-DIR pos-1-2 pos-1-1 dir-up)
    (MOVE-DIR pos-1-2 pos-1-3 dir-down)
    (MOVE-DIR pos-1-2 pos-2-2 dir-right)
    (MOVE-DIR pos-1-3 pos-1-2 dir-up)
    (MOVE-DIR pos-1-3 pos-2-3 dir-right)
    
    (MOVE-DIR pos-2-1 pos-1-1 dir-left)
    (MOVE-DIR pos-2-1 pos-2-2 dir-down)
    (MOVE-DIR pos-2-1 pos-3-1 dir-right)
    (MOVE-DIR pos-2-2 pos-1-2 dir-left)
    (MOVE-DIR pos-2-2 pos-2-1 dir-up)
    (MOVE-DIR pos-2-2 pos-2-3 dir-down)
    (MOVE-DIR pos-2-2 pos-3-2 dir-right)
    (MOVE-DIR pos-2-3 pos-1-3 dir-left)
    (MOVE-DIR pos-2-3 pos-2-2 dir-up)
    (MOVE-DIR pos-2-3 pos-3-3 dir-right)
    
    (MOVE-DIR pos-3-1 pos-2-1 dir-left)
    (MOVE-DIR pos-3-1 pos-3-2 dir-down)
    (MOVE-DIR pos-3-2 pos-2-2 dir-left)
    (MOVE-DIR pos-3-2 pos-3-1 dir-up)
    (MOVE-DIR pos-3-2 pos-3-3 dir-down)
    (MOVE-DIR pos-3-3 pos-2-3 dir-left)
    (MOVE-DIR pos-3-3 pos-3-2 dir-up)
    
    ;; Initial state
    (at player-01 pos-1-1)
    (at stone-01 pos-2-2)
    
    ;; Clear locations
    (clear pos-1-2)
    (clear pos-1-3)
    (clear pos-2-1)
    (clear pos-2-3)
    (clear pos-3-1)
    (clear pos-3-2)
    (clear pos-3-3)
    
    ;; Initial cost
    (= (total-cost) 0)
  )
  (:goal (and
    (at-goal stone-01)
  ))
  (:metric minimize (total-cost))
)