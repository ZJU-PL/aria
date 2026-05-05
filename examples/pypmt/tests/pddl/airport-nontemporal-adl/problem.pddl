(define (problem VERY_SIMPLE_AIRPORT)
(:domain airport)

(:objects
      ;; the airplane
      airplane_A - airplane

      ;; the airplanetype
      medium - airplanetype

      ;; the directions
      north
      south - direction

      ;; just two segments
      seg_runway
      seg_gate - segment
)

(:init
      ;; Initial position of airplane
      (at-segment airplane_A seg_runway)
      (blocked seg_runway airplane_A)

      ;; Movement capability
      (can-move seg_runway seg_gate south)
      
      ;; Taxiing direction
      (move-dir seg_runway seg_gate south)
      
      ;; Aircraft properties
      (facing airplane_A south)
      (has-type airplane_A medium)
      (is-moving airplane_A)
      
      ;; Blocking relationship
      (is-blocked seg_gate medium seg_runway south)
      
      ;; Runway property
      (is-start-runway seg_runway north)
      
      ;; Occupied segment
      (occupied seg_runway)
)

(:goal
      (and
            (is-parked airplane_A seg_gate)
      )
)
)