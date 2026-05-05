(define (problem simple-logistics)
   (:domain logistics-adl)
   (:objects 
             package1 package2 - obj
             city1 city2 - city
             truck1 truck2 - truck
             plane1 - airplane
             city1-1 city2-1 - location
             city1-2 city2-2 - airport)
   (:init 
          (in-city city1-1 city1)
          (in-city city1-2 city1)
          (in-city city2-1 city2)
          (in-city city2-2 city2)
          
          (at plane1 city1-2)
          (at truck1 city1-1)
          (at truck2 city2-1)
          
          (at package1 city1-1)
          (at package2 city2-1))
          
   (:goal (and 
               (at package1 city2-1)
               (at package2 city1-2))))