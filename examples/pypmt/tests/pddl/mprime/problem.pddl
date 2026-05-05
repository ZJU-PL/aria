(define (problem mprime-x-1)
   (:domain mystery-prime-typed)
   (:objects rice pear flounder okra pork lamb - food
             rest - pleasure
             hangover depression abrasion - pain
)
   (:init
(eats lamb pork)
          (eats pork okra)
(= (locale okra) 6)
(= (locale pork) 5)
          (eats pork lamb)
(= (harmony rest) 3)
          (eats lamb flounder)
          (craves depression flounder)
          (eats okra pear)
          (eats rice rice)
          (eats rice flounder)



          (craves abrasion pork)
(= (locale rice) 1)

          (eats flounder lamb)
          (craves rest pork)

(= (locale pear) 2)
          (craves hangover rice)


(= (locale lamb) 3)
          (eats flounder rice)
          (eats rice pear)

          (eats pear okra)
          (eats pear rice)
          (eats okra pork)

(= (locale flounder) 4))
   (:goal (and (craves abrasion rice))))
