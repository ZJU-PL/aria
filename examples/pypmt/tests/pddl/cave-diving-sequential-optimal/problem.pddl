(define (problem minimal-cave-diving)
  (:domain cave-diving-adl)
  
  (:objects
    l0 l1 - location
    d0 - diver
    t0 t1 t2 t3 dummy - tank
    zero one two three - quantity
  )

  (:init
    (available d0)
    (capacity d0 three)
    (in-storage t0)
    (next-tank t0 t1)
    (next-tank t1 t2)
    (next-tank t2 t3)
    (next-tank t3 dummy)
    (cave-entrance l0)
    (connected l0 l1)
    (connected l1 l0)
    (next-quantity zero one)
    (next-quantity one two)
    (next-quantity two three)
    (= (hiring-cost d0) 10)
    (= (other-cost) 1)
    (= (total-cost) 0)
  )

  (:goal
    (and
      (have-photo l1)
      (decompressing d0)
    )
  )

  (:metric minimize (total-cost))
)