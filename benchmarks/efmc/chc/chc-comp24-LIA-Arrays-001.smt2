(set-logic HORN)


(declare-fun |hanoi| ( Bool Bool Bool (Array Int Int) (Array Int Int) Int Int ) Bool)
(declare-fun |hanoi@_1| ( (Array Int Int) Int ) Bool)
(declare-fun |hanoi@UnifiedReturnBlock.split| ( (Array Int Int) (Array Int Int) Int Int ) Bool)
(declare-fun |applyHanoi@tailrecurse| ( (Array Int Int) (Array Int Int) Int Int Int ) Bool)
(declare-fun |applyHanoi@tailrecurse._crit_edge| ( (Array Int Int) (Array Int Int) Int Int ) Bool)
(declare-fun |main@entry.split| ( ) Bool)
(declare-fun |main@entry| ( (Array Int Int) ) Bool)
(declare-fun |applyHanoi@_1| ( (Array Int Int) Int Int ) Bool)
(declare-fun |applyHanoi| ( Bool Bool Bool (Array Int Int) (Array Int Int) Int Int ) Bool)

(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (and true (= v_4 true) (= v_5 true) (= v_6 true))
      )
      (applyHanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (and true (= v_4 false) (= v_5 true) (= v_6 true))
      )
      (applyHanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (and true (= v_4 false) (= v_5 false) (= v_6 false))
      )
      (applyHanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (applyHanoi@tailrecurse._crit_edge A B D C)
        (and (= v_4 true) (= v_5 false) (= v_6 false))
      )
      (applyHanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B Int) (C Int) )
    (=>
      (and
        true
      )
      (applyHanoi@_1 A B C)
    )
  )
)
(assert
  (forall ( (A Bool) (B Bool) (C (Array Int Int)) (D Bool) (E Bool) (F Int) (G (Array Int Int)) (H (Array Int Int)) (I Int) (J Int) (K Int) )
    (=>
      (and
        (applyHanoi@_1 G I K)
        (and (or (not E) (not B) (not A))
     (or (not E) (not D) (= C G))
     (or (not E) (not D) (= H C))
     (or (not E) (not D) (= F K))
     (or (not E) (not D) (= J F))
     (or (not D) (and E D))
     (or (not E) (and E A))
     (= D true)
     (= B (= K 0)))
      )
      (applyHanoi@tailrecurse G H I J K)
    )
  )
)
(assert
  (forall ( (A Int) (B (Array Int Int)) (C Int) (D Int) (E (Array Int Int)) (F Bool) (G (Array Int Int)) (H Int) (I (Array Int Int)) (J Bool) (K Bool) (L Int) (M (Array Int Int)) (N (Array Int Int)) (O Int) (P Int) (Q Int) (v_17 Bool) (v_18 Bool) (v_19 Bool) )
    (=>
      (and
        (applyHanoi@tailrecurse M B O D Q)
        (applyHanoi v_17 v_18 v_19 E G H O)
        (and (= v_17 true)
     (= v_18 false)
     (= v_19 false)
     (= F (= H 0))
     (= A (select B O))
     (= C (+ 1 A))
     (= H (+ (- 1) D))
     (or (not K) (not J) (= I G))
     (or (not K) (not J) (= N I))
     (or (not K) (not J) (= L H))
     (or (not K) (not J) (= P L))
     (or (not K) (not J) (not F))
     (or (not J) (and K J))
     (= J true)
     (= E (store B O C)))
      )
      (applyHanoi@tailrecurse M N O P Q)
    )
  )
)
(assert
  (forall ( (A Bool) (B Bool) (C Bool) (D (Array Int Int)) (E (Array Int Int)) (F (Array Int Int)) (G Int) (H Int) )
    (=>
      (and
        (applyHanoi@_1 E G H)
        (and (or (not C) (not B) (= D E))
     (or (not C) (not B) (= F D))
     (or (not C) (not B) A)
     (or (not B) (and C B))
     (= B true)
     (= A (= H 0)))
      )
      (applyHanoi@tailrecurse._crit_edge E F G H)
    )
  )
)
(assert
  (forall ( (A Int) (B (Array Int Int)) (C Int) (D Int) (E (Array Int Int)) (F Int) (G Bool) (H Bool) (I (Array Int Int)) (J Bool) (K Bool) (L (Array Int Int)) (M (Array Int Int)) (N (Array Int Int)) (O Int) (P Int) (v_16 Bool) (v_17 Bool) (v_18 Bool) )
    (=>
      (and
        (applyHanoi@tailrecurse M B O D P)
        (applyHanoi v_16 v_17 v_18 E I F O)
        (and (= v_16 true)
     (= v_17 false)
     (= v_18 false)
     (= H (= F 0))
     (= A (select B O))
     (= F (+ (- 1) D))
     (= C (+ 1 A))
     (or (not K) (not G) H)
     (or (not K) (not J) (= L I))
     (or (not K) (not J) (= N L))
     (or (not J) (and K J))
     (or (not K) (and K G))
     (= J true)
     (= E (store B O C)))
      )
      (applyHanoi@tailrecurse._crit_edge M N O P)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (and true (= v_4 true) (= v_5 true) (= v_6 true))
      )
      (hanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (and true (= v_4 false) (= v_5 true) (= v_6 true))
      )
      (hanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (and true (= v_4 false) (= v_5 false) (= v_6 false))
      )
      (hanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B (Array Int Int)) (C Int) (D Int) (v_4 Bool) (v_5 Bool) (v_6 Bool) )
    (=>
      (and
        (hanoi@UnifiedReturnBlock.split A B D C)
        (and (= v_4 true) (= v_5 false) (= v_6 false))
      )
      (hanoi v_4 v_5 v_6 A B C D)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B Int) )
    (=>
      (and
        true
      )
      (hanoi@_1 A B)
    )
  )
)
(assert
  (forall ( (A Int) (B Int) (C Int) (D Bool) (E Bool) (F (Array Int Int)) (G Int) (H (Array Int Int)) (I Bool) (J Int) (K (Array Int Int)) (L Bool) (M Int) (N Bool) (O Bool) (P (Array Int Int)) (Q (Array Int Int)) (R Int) (S Int) (v_19 Bool) (v_20 Bool) )
    (=>
      (and
        (hanoi@_1 P S)
        (hanoi L v_19 v_20 P F A B)
        (and (= v_19 false)
     (= v_20 false)
     (or (not I) E (not D))
     (or (not L) (not E) (not D))
     (or (not N) (and N L) (and N I))
     (or (not N) (not I) (= H P))
     (or (not N) (not I) (= Q H))
     (or (not N) (not I) (= J 1))
     (or (not N) (not I) (= R J))
     (or (not N) (not L) (= K F))
     (or (not N) (not L) (= Q K))
     (or (not N) (not L) (= M G))
     (or (not N) (not L) (= R M))
     (or (not O) (and N O))
     (or (not I) (and I D))
     (or (not L) (= A (+ (- 1) S)))
     (or (not L) (= G (+ 1 C)))
     (or (not L) (= C (* 2 B)))
     (or (not L) (and L D))
     (= O true)
     (= E (= S 1)))
      )
      (hanoi@UnifiedReturnBlock.split P Q R S)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) )
    (=>
      (and
        true
      )
      (main@entry A)
    )
  )
)
(assert
  (forall ( (A (Array Int Int)) (B Int) (C Bool) (D (Array Int Int)) (E (Array Int Int)) (F (Array Int Int)) (G Int) (H (Array Int Int)) (I Int) (J Int) (K Int) (L Bool) (M Bool) (N Bool) (v_14 Bool) (v_15 Bool) (v_16 Bool) (v_17 Bool) (v_18 Bool) (v_19 Bool) )
    (=>
      (and
        (main@entry A)
        (applyHanoi v_14 v_15 v_16 E F G I)
        (hanoi v_17 v_18 v_19 F H G J)
        (let ((a!1 (= C (or (not (<= B 30)) (not (>= B 0))))))
  (and (= v_14 true)
       (= v_15 false)
       (= v_16 false)
       (= v_17 true)
       (= v_18 false)
       (= v_19 false)
       (= D (store A I 0))
       (= L (= J K))
       a!1
       (= K (select H I))
       (= B (+ (- 1) G))
       (or (not N) (and N M))
       (not L)
       (= N true)
       (not C)
       (= E (store D I 0))))
      )
      main@entry.split
    )
  )
)
(assert
  (forall ( (CHC_COMP_UNUSED Bool) )
    (=>
      (and
        main@entry.split
        true
      )
      false
    )
  )
)

(check-sat)
(exit)