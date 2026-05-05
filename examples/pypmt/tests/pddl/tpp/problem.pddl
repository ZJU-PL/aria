;; Simplified TPP Problem
(define (problem simple-tpp)
(:domain TPP-Metric)
(:objects
	market1 market2 - market
	depot0 - depot
	truck0 - truck
	goods0 - goods)
(:init
	;; Market prices and availability
	(= (price goods0 market1) 10)
	(= (on-sale goods0 market1) 5)
	(= (price goods0 market2) 15)
	(= (on-sale goods0 market2) 10)
	
	;; Truck location
	(loc truck0 depot0)
	
	;; Drive costs
	(= (drive-cost depot0 market1) 50)
	(= (drive-cost market1 depot0) 50)
	(= (drive-cost depot0 market2) 100)
	(= (drive-cost market2 depot0) 100)
	(= (drive-cost market1 market2) 75)
	(= (drive-cost market2 market1) 75)
	
	;; Initial state
	(= (bought goods0) 0)
	(= (request goods0) 7)
	(= (total-cost) 0))

(:goal (and
	(>= (bought goods0) (request goods0))
	(loc truck0 depot0)))

(:metric minimize (total-cost))
)