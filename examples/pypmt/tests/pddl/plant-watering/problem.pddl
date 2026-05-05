(define (problem instance_4_1)
  (:domain mt-plant-watering-constrained)
  (:objects
    tap1 - tap
	agent1 - agent
	plant1 - plant
  )

  (:init
    (= (max_int) 80)
	(= (maxx) 4)
	(= (minx) 1)
	(= (maxy) 4)
	(= (miny) 1)
	(= (carrying) 10)
	(= (total_poured) 0)
	(= (total_loaded) 0)
	(= (poured plant1) 0)
	(= (x agent1) 3)
	(= (y agent1) 1)

	(= (x plant1) 2)
	(= (y plant1) 2)

	(= (x tap1) 3)
	(= (y tap1) 3)

	(= (total-cost) 0)
  )

  (:goal (and 
    (= (poured plant1) 4)
  ))

  (:metric minimize (total-cost))
  

  
)
