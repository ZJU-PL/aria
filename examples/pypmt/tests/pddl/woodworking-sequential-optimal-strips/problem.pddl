; Simplified woodworking task with just 1 part
; Machines:
;   1 grinder
;   1 immersion-varnisher
;   1 highspeed-saw
; random seed: 123456

(define (problem simple-wood-prob)
  (:domain woodworking)
  (:objects
    grinder0 - grinder
    immersion-varnisher0 - immersion-varnisher
    highspeed-saw0 - highspeed-saw
    mauve - acolour
    beech - awood
    p0 - part
    b0 - board
    s0 s1 s2 - aboardsize
  )
  (:init
    (grind-treatment-change varnished colourfragments)
    (grind-treatment-change glazed untreated)
    (grind-treatment-change untreated untreated)
    (grind-treatment-change colourfragments untreated)
    (is-smooth smooth)
    (is-smooth verysmooth)
    (= (total-cost) 0)
    (boardsize-successor s0 s1)
    (boardsize-successor s1 s2)
    (has-colour immersion-varnisher0 mauve)
    (empty highspeed-saw0)
    (unused p0)
    (goalsize p0 small)
    (= (spray-varnish-cost p0) 5)
    (= (glaze-cost p0) 10)
    (= (grind-cost p0) 15)
    (= (plane-cost p0) 10)
    (boardsize b0 s2)
    (wood b0 beech)
    (surface-condition b0 smooth)
    (available b0)
  )
  (:goal
    (and
      (available p0)
      (colour p0 mauve)
      (wood p0 beech)
      (treatment p0 varnished)
    )
  )
  (:metric minimize (total-cost))
)