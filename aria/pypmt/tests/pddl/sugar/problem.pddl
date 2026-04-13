(define (problem simple-supply-chain)
	(:domain supply-chain)
	
	(:objects
		brand1 brand2 - brand
		sugar-cane    - raw-cane
		truck1        - truck
		depot1        - depot
		mill1         - mill
		crane1        - crane
	)

	(:init
		(=(unharvest-field)2) (=(mill-cost)0) (=(inventory-cost)0) (=(handling-cost)0)
		(=(cost-process mill1)1)
		(=(has-resource sugar-cane mill1)5)
		(=(max-changing mill1)2)
		(=(max-produce mill1)3)
		(available mill1)

		(produce mill1 brand1) (produce mill1 brand2) (current-process mill1 brand1)
		(=(in-storage mill1 brand1)0) (=(in-storage mill1 brand2)0)

		(change-process brand1 brand2)
		(change-process brand2 brand1)
		
		(at-location truck1 mill1)
		(=(truck-cap truck1)5)
		(at-location crane1 mill1)
		(ready-crane crane1)
		(=(capacity crane1)2)
		(=(service-time crane1)5)
		(=(max-service-time crane1)5)
		(=(in-truck-sugar brand1 truck1)0)
		(=(in-truck-sugar brand2 truck1)0)
		
		(=(in-storage depot1 brand1)0) 
		(=(in-storage depot1 brand2)0)

		(connected mill1 depot1) (connected depot1 mill1)
	)
	(:goal (and
		 (>=(in-storage depot1 brand1)2)
		)
	)
)