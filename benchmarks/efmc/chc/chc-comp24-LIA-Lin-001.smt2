(set-logic HORN)


(declare-fun |write| ( Int Int Int Int ) Bool)
(declare-fun |CHC_COMP_FALSE| ( ) Bool)
(declare-fun |incr| ( Int Int Int Int ) Bool)
(declare-fun |end| ( Int Int Int ) Bool)
(declare-fun |loop| ( Int Int Int Int ) Bool)

(assert
  (forall ( (A Int) (B Int) (C Int) (v_3 Int) ) 
    (=>
      (and
        (and (not (<= A B)) (<= 0 B) (= 0 v_3))
      )
      (loop A v_3 B C)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) ) 
    (=>
      (and
        (loop A B C D)
        (not (<= A B))
      )
      (write A B C D)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) ) 
    (=>
      (and
        (write A B C D)
        (not (= B C))
      )
      (incr A B C D)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) (E Int) (v_5 Int) ) 
    (=>
      (and
        (write B C D E)
        (and (= A (mod C 2)) (= v_5 C))
      )
      (incr B C v_5 A)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) (E Int) ) 
    (=>
      (and
        (incr B C D E)
        (= A (+ 1 C))
      )
      (loop B A D E)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) ) 
    (=>
      (and
        (loop A B C D)
        (>= B A)
      )
      (end A C D)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) ) 
    (=>
      (and
        (end B A D)
        (and (not (= D 0)) (= A (* 2 C)))
      )
      CHC_COMP_FALSE
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Int) ) 
    (=>
      (and
        (end B A D)
        (and (not (= D 1)) (= A (+ 1 (* 2 C))))
      )
      CHC_COMP_FALSE
    )
  )
)
(assert
  (forall ( (CHC_COMP_UNUSED Bool) ) 
    (=>
      (and
        CHC_COMP_FALSE
      )
      false
    )
  )
)

(check-sat)
(exit)