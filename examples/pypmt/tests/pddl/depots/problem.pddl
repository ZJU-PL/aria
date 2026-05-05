(define (problem depotprob1818) (:domain depot)
(:objects
	depot0 - depot
	distributor0 distributor1 - distributor
	truck0 truck1 - truck
	pallet0 pallet1 pallet2 - pallet
	crate0 crate1 - crate
	hoist0 hoist1 hoist2 - hoist)
(:init
	(located pallet0 depot0)
	(clear crate1)
	(located pallet1 distributor0)
	(clear crate0)
	(located pallet2 distributor1)
	(clear pallet2)
	(located truck0 distributor1)
	(= (current_load truck0) 0)
	(= (load_limit truck0) 323)
	(located truck1 depot0)
	(= (current_load truck1) 0)
	(= (load_limit truck1) 220)
	(located hoist0 depot0)
	(available hoist0)
	(located hoist1 distributor0)
	(available hoist1)
	(located hoist2 distributor1)
	(available hoist2)
	(located crate0 distributor0)
	(on crate0 pallet1)
	(= (weight crate0) 11)
	(located crate1 depot0)
	(on crate1 pallet0)
	(= (weight crate1) 86)
	(= (fuel-cost) 0)
)

(:goal (and
		(on crate0 pallet2)
		(on crate1 pallet1)
	)
)

(:metric minimize (fuel-cost)))
