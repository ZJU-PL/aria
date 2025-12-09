(define-fun insert ((x Int) (y (Array Int Bool))) (Array Int Bool)
  (store y x true)
)

(define-fun member ((x Int) (y (Array Int Bool))) Bool
  (select y x)
)

(define-fun empIntSet () (Array Int Bool)
  ((as const (Array Int Bool)) false)
)
