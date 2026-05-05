(define (problem instance_4_1)
  (:domain ext-plant-watering)
  (:objects
    tap1 - tap
	agent1 - agent
	plant1 plant2 plant3 plant4 - plant
  )

  (:init
	(= (maxx) 4)
	(= (minx) 1)
	(= (maxy) 4)
	(= (miny) 1)
	(= (total_poured) 0)
	(= (total_loaded) 0)
	(= (water_reserve) 25) ; only 22 out of 25 needed

    (= (x tap1) 3)
	(= (y tap1) 3)
	
	(= (poured plant1) 0)

	(= (x agent1) 3)
	(= (y agent1) 1)
	(= (carrying agent1) 0)
    (= (max_carry agent1) 5)

    (= (x plant1) 2)
	(= (y plant1) 2)
  )

  (:goal (and 
    (= (poured plant1) 4)
    ; makes sure all agents end empty
	(= (total_poured) (total_loaded)) 
  ))
)
