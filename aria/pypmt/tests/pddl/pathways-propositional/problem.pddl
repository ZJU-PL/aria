(define (problem Simple-Pathways)
(:domain Pathways-Propositional)
(:objects
	SP1 - simple
	E2F13 - simple
	pCAF - simple
	p300 - simple
	SP1-E2F13 - complex
	l0 - level
	l1 - level
	l2 - level)


(:init
	(possible SP1)
	(possible E2F13)
	(possible pCAF)
	(possible p300)
	(association-reaction SP1 E2F13 SP1-E2F13)
	(association-reaction pCAF p300 pCAF-p300)
	(num-subs l0)
	(next l1 l0)
	(next l2 l1))


(:goal
	(and
	(goal1)))

)