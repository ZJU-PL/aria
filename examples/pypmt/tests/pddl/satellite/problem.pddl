(define (problem simple-satellite)
(:domain satellite)
(:objects
  satellite0 - satellite
  instrument0 - instrument
  thermograph0 - mode
  GroundStation - direction
  Target1 - direction
  Target2 - direction
)
(:init
  ;; Instrument capabilities
  (supports instrument0 thermograph0)
  (calibration_target instrument0 GroundStation)
  (on_board instrument0 satellite0)
  
  ;; Initial satellite state
  (power_avail satellite0)
  (pointing satellite0 GroundStation)
  
  ;; Resource capacities
  (= (data_capacity satellite0) 100)
  (= (fuel satellite0) 50)
  
  ;; Data sizes for images
  (= (data Target1 thermograph0) 25)
  (= (data Target2 thermograph0) 30)
  
  ;; Slew times (movement costs)
  (= (slew_time GroundStation Target1) 10)
  (= (slew_time Target1 GroundStation) 10)
  (= (slew_time GroundStation Target2) 15)
  (= (slew_time Target2 GroundStation) 15)
  (= (slew_time Target1 Target2) 5)
  (= (slew_time Target2 Target1) 5)
  
  ;; Initial counters
  (= (data-stored) 0)
  (= (fuel-used) 0)
)
(:goal (and
  (have_image Target1 thermograph0)
  (have_image Target2 thermograph0)
))
(:metric minimize (fuel-used))
)