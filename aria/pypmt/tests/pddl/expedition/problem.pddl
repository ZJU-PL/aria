;; Ben Pathak (pathak.ban@gmail.com)
(define (problem instance_2_sled_1)

	(:domain expedition)

	(:objects
		s0 s1 - sled
		wa0 wa1 wa2 wb0 wb1 wb2 - waypoint
	)

  (:init
		(at s0 wa0)
		(= (sled_capacity s0) 4)
		(= (sled_supplies s0) 1)
		(= (waypoint_supplies wa0) 1000)
		(= (waypoint_supplies wa1) 0)
		(= (waypoint_supplies wa2) 0)
		(is_next wa0 wa1)
		(is_next wa1 wa2)
		(at s1 wb0)
		(= (sled_capacity s1) 4)
		(= (sled_supplies s1) 1)
		(= (waypoint_supplies wb0) 1000)
		(= (waypoint_supplies wb1) 0)
		(= (waypoint_supplies wb2) 0)
		(is_next wb0 wb1)
		(is_next wb1 wb2)
	)

	(:goal
		(and
			(at s0 wa2)
			(at s1 wb2)
		)
	)
)


