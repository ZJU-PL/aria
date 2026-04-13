(define (problem simple-openstacks)
(:domain openstacks-sequencedstrips-ADL)
(:objects 
n0 n1 n2 n3  - count
o1 o2  - order
p1 p2  - product
)

(:init
(next-count n0 n1) (next-count n1 n2) (next-count n2 n3)
(stacks-avail n0)

(waiting o1)
(includes o1 p1)

(waiting o2)
(includes o2 p2)

(= (total-cost) 0)
)

(:goal
(and
(shipped o1)
(shipped o2)
))

(:metric minimize (total-cost))
)