(define (problem simple-floor-tile)
 (:domain floor-tile)
 (:objects 
           tile_0-1 tile_0-2 
           tile_1-1 tile_1-2 - tile
           robot1 - robot
           white black - color
)
 (:init 
   (= (total-cost) 0)
   (robot-at robot1 tile_0-1)
   (robot-has robot1 white)
   (available-color white)
   (available-color black)
   (clear tile_0-2)
   (clear tile_1-1)
   (clear tile_1-2)
   (up tile_1-1 tile_0-1)
   (up tile_1-2 tile_0-2)
   (down tile_0-1 tile_1-1)
   (down tile_0-2 tile_1-2)
   (right tile_0-2 tile_0-1)
   (right tile_1-2 tile_1-1)
   (left tile_0-1 tile_0-2)
   (left tile_1-1 tile_1-2)
)
 (:goal (and
    (painted tile_1-1 white)
    (painted tile_1-2 black)
))
 (:metric minimize (total-cost))
)