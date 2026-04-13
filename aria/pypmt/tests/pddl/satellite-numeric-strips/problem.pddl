(define (problem mini-satellite)
(:domain satellite)
(:objects
  satellite0 - satellite
  instrument0 - instrument
  thermograph0 - mode
  GroundStation - direction
  Target1 - direction
)
(:init
  ;; Instrument capabilities
  (supports instrument0 thermograph0)
  (calibration_target instrument0 GroundStation)
  (on_board instrument0 satellite0)
  
  ;; Initial satellite state
  (power_avail satellite0)
  (pointing satellite0 Target1)
  
  ;; Resource capacities
  (= (data_capacity satellite0) 50)
  (= (fuel satellite0) 20)
  
  ;; Data sizes for images
  (= (data Target1 thermograph0) 10)
  
  ;; Slew times (movement costs)
  (= (slew_time GroundStation Target1) 5)
  (= (slew_time Target1 GroundStation) 5)
  
  ;; Initial counters
  (= (data-stored) 0)
  (= (fuel-used) 0)
)
(:goal (and
  (have_image Target1 thermograph0)
))
(:metric minimize (fuel-used))
)