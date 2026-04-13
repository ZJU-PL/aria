(define (problem Simple-Pathways-Metric)
(:domain Pathways-Metric)
(:objects
	pCAF - simple
	p300 - simple
	pCAF-p300 - complex)

(:init
	(possible pCAF)
	(= (available pCAF) 0)
	(possible p300)
	(= (available p300) 0)
	(= (available pCAF-p300) 0)
	
	(association-reaction pCAF p300 pCAF-p300)
	(= (need-for-association pCAF p300 pCAF-p300) 1)
	(= (need-for-association p300 pCAF pCAF-p300) 1)
	(= (prod-by-association pCAF p300 pCAF-p300) 1)
	
	(= (num-subs) 0)
	(= (duration-association-reaction pCAF p300 pCAF-p300) 1.0))

(:goal
	(and
	(>= (available pCAF-p300) 1)))
)