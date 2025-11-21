(set-info :smt-lib-version 2.6)
(set-logic QF_BV)
(set-info :category "industrial")
(declare-fun k!219 () (_ BitVec 8))
(declare-fun k!218 () (_ BitVec 8))
(declare-fun k!217 () (_ BitVec 8))
(declare-fun k!216 () (_ BitVec 8))
(declare-fun k!228 () (_ BitVec 8))
(declare-fun k!229 () (_ BitVec 8))
(declare-fun k!230 () (_ BitVec 8))
(declare-fun k!231 () (_ BitVec 8))
(declare-fun k!232 () (_ BitVec 8))
(declare-fun k!233 () (_ BitVec 8))
(declare-fun k!234 () (_ BitVec 8))
(declare-fun k!235 () (_ BitVec 8))
(declare-fun k!28 () (_ BitVec 8))
(declare-fun k!29 () (_ BitVec 8))
(declare-fun k!30 () (_ BitVec 8))
(declare-fun k!31 () (_ BitVec 8))
(declare-fun k!44 () (_ BitVec 8))
(declare-fun k!45 () (_ BitVec 8))
(assert (let ((a!1 (bvadd (concat #b000000000000000000000000000
                          (bvadd #xffffffff (concat #x0000 k!45 k!44))
                          #b00000)
                  (concat #x00000000 k!31 k!30 k!29 k!28)))
      (a!2 (bvadd (concat (bvadd #b111111111111111111111111111
                                 (concat #b00000000000 k!45 k!44))
                          #b00000)
                  (concat k!31 k!30 k!29 k!28)))
      (a!4 (= (concat #b000000000000000000000000000
                      (bvadd #xffffffff (concat #x0000 k!45 k!44))
                      #b00000)
              (bvadd #x0000000000000134
                     (bvmul #xffffffffffffffff
                            (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!5 (= (concat #b000000000000000000000000000
                      (bvadd #xffffffff (concat #x0000 k!45 k!44))
                      #b00000)
              (bvadd #xffffffffffffffff
                     (bvmul #xffffffffffffffff
                            (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!6 (bvmul #xffffffffffffffff
                  (concat #b000000000000000000000000000
                          (bvadd #xffffffff (concat #x0000 k!45 k!44))
                          #b00000)))
      (a!8 (or (bvsle (concat #b00000000000000000000000000011
                              ((_ extract 2 0) k!28))
                      #xffffffff)
               (bvsle #x00000001
                      (concat #b00000000000000000000000000011
                              ((_ extract 2 0) k!28)))))
      (a!9 (or (bvsle (concat #b00000 ((_ extract 2 2) k!28) #b00) #xff)
               (bvsle #x01 (concat #b00000 ((_ extract 2 2) k!28) #b00))))
      (a!10 (or (bvule (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28))
                       #xfffffffffffffe33)
                (bvule #xfffffffffffffe74
                       (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!11 (or (bvule (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28))
                       #xfffffffffffffe73)
                (bvule #xfffffffffffffeb4
                       (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!12 (or (bvule (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28))
                       #xfffffffffffffeb3)
                (bvule #xfffffffffffffef4
                       (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!13 (or (bvule (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28))
                       #xfffffffffffffef3)
                (bvule #xffffffffffffff34
                       (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!14 (or (bvule (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28))
                       #xffffffffffffff33)
                (bvule #xffffffffffffff54
                       (bvmul #xffffffffffffffff
                              (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!15 (and (bvsle #x00000001
                        (concat #b00000000000000000000000000000
                                ((_ extract 2 0) k!28)))
                 (bvsle (concat #b00000000000000000000000000000
                                ((_ extract 2 0) k!28))
                        #x00000006)))
      (a!17 (bvule #x00000008
                   (bvadd #x00000300
                          (bvmul #xffffffff
                                 (concat #b0000000000 k!45 k!44 #b000000)))))
      (a!18 (or (bvule (bvmul #xffffffffffffffff
                              (concat #b000000000000000000000000000000000000000000
                                      k!45
                                      k!44
                                      #b000000))
                       #xffff80021edcb325)
                (bvule #xffff80021edcb332
                       (bvmul #xffffffffffffffff
                              (concat #b000000000000000000000000000000000000000000
                                      k!45
                                      k!44
                                      #b000000)))))
      (a!19 (= ((_ extract 63 4)
                 (bvadd #x00000000011f7fc0
                        (concat #b000000000000000000000000000000000000000000
                                k!45
                                k!44
                                #b000000)))
               #x00000000011f820))
      (a!20 ((_ extract 31 8)
              (bvadd #x000002f8
                     (bvmul #xffffffff (concat #b0000000000 k!45 k!44 #b000000)))))
      (a!21 (bvadd #xf8 (bvmul #xff (concat ((_ extract 1 0) k!44) #b000000))))
      (a!22 (not (= (concat #x00000000 k!219 k!218 k!217 k!216)
                    (bvadd #x0000000000000120
                           (concat #x00000000 k!31 k!30 k!29 k!28)))))
      (a!23 (and (= k!228 #x00) (= k!229 #x00) (= k!230 #x00) (= k!231 #x00)))
      (a!24 (bvule (concat k!235 k!234 k!233 k!232)
                   (concat k!231 k!230 k!229 k!228)))
      (a!26 (= ((_ extract 63 4)
                 (bvadd #x0000000000000018
                        (concat #x00000000 k!231 k!230 k!229 k!228)))
               #x000000000000007))
      (a!27 (= ((_ extract 63 8)
                 (bvadd #x0000000000000018
                        (concat #x00000000 k!231 k!230 k!229 k!228)))
               #x00000000000000))
      (a!28 (concat ((_ extract 7 4)
                      (bvadd #x0000000000000018
                             (concat #x00000000 k!231 k!230 k!229 k!228)))
                    #x0))
      (a!29 (concat #x0
                    ((_ extract 31 4)
                      (bvadd #x0000000000000018
                             (concat #x00000000 k!231 k!230 k!229 k!228)))))
      (a!33 (bvadd #b00001
                   ((_ extract 8 4)
                     (bvadd #x0000000000000018
                            (concat #x00000000 k!231 k!230 k!229 k!228)))))
      (a!39 (bvule (concat #x00000000 k!231 k!230 k!229 k!228)
                   (bvadd #x00000000000001cc
                          (bvmul #xffffffffffffffff
                                 (concat #x00000000 k!219 k!218 k!217 k!216)))))
      (a!40 (or (bvsle (concat #b00000
                               k!231
                               k!230
                               k!229
                               ((_ extract 7 5) k!228))
                       #xffffffff)
                (bvsle #x00000003
                       (concat #b00000
                               k!231
                               k!230
                               k!229
                               ((_ extract 7 5) k!228))))))
(let ((a!3 (not (and (= ((_ extract 63 32) a!1) #x00000000)
                     (bvule a!2 (concat k!31 k!30 k!29 k!28)))))
      (a!7 (bvadd a!6
                  (bvmul #xffffffffffffffff
                         (concat #x00000000 k!31 k!30 k!29 k!28))))
      (a!16 (or (bvsle (concat #b00000000000000000000000000000
                               ((_ extract 2 0) k!28))
                       #xffffffff)
                a!15
                (bvsle #x00000008
                       (concat #b00000000000000000000000000000
                               ((_ extract 2 0) k!28)))))
      (a!25 (bvor (bvnot (ite a!23 #x00 #x01))
                  (bvnot (bvmul #xff (ite a!24 #x00 #x01)))))
      (a!30 (= (bvmul #x0000000000000008
                      (concat #x00000000 (bvadd #xfffffffe a!29)))
               #x0000000000000028))
      (a!31 (bvmul #x0000000000000008
                   (concat #x00000000
                           (bvadd #xfffffffe (bvmul #x00000002 a!29)))))
      (a!32 (bvmul #x0000000000000004
                   (concat #b0000000000000000000000000000000000000
                           ((_ extract 31 5) (bvadd #x00000001 a!29)))))
      (a!34 (bvadd #x00000001
                   (concat #b00000 ((_ extract 31 5) (bvadd #x00000001 a!29)))))
      (a!35 (bvadd #x00000002
                   (concat #b00000 ((_ extract 31 5) (bvadd #x00000001 a!29)))))
      (a!36 (bvsle (concat #b00000 ((_ extract 31 5) (bvadd #x00000001 a!29)))
                   #x00000000))
      (a!37 (bvsle #x00000002
                   (concat #b00000 ((_ extract 31 5) (bvadd #x00000001 a!29)))))
      (a!38 (or (not (bvsle #x0000000000000000
                            (concat #x00000000 k!231 k!230 k!229 k!228)))
                a!23)))
  (and (or false
           (and (bvuge k!233 #x00) (bvule k!233 #x00))
           (and (bvuge k!234 #x00) (bvule k!234 #x00))
           (and (bvuge k!235 #x00) (bvule k!235 #x00)))
       (or (bvsle (concat #x00000000 k!31 k!30 k!29 k!28) #xffffffffffffffff)
           (bvsle #x0000000000000001 (concat #x00000000 k!31 k!30 k!29 k!28)))
       (or (bvsle (concat #x0000 k!45 k!44) #x0000fffe)
           (bvsle #x00010000 (concat #x0000 k!45 k!44)))
       (bvule #x0002 (concat k!45 k!44))
       a!3
       a!4
       (= (ite a!5 #x01 #x00) #x00)
       (or (bvule a!7 #xfffffffffffffe33) (bvule #xfffffffffffffe54 a!7))
       (= ((_ extract 2 0) k!28) #b100)
       (bvule (concat #b0000000 ((_ extract 2 0) k!28)) #b1111101000)
       a!8
       (= ((_ extract 0 0) k!28) #b0)
       (= ((_ extract 1 1) k!28) #b0)
       a!9
       (bvsle #x0000000000000000 (concat #x00000000 k!31 k!30 k!29 k!28))
       (bvsle (concat #x00000000 k!31 k!30 k!29 k!28) #x00000000000001cb)
       (bvsle #x0000000000000001
              (concat #b000000000000000000000000000000000000000000
                      k!45
                      k!44
                      #b000000))
       (= k!44 #x09)
       (= k!45 #x00)
       (= ((_ extract 7 4) k!44) #x0)
       (bvule (concat ((_ extract 3 0) k!44) #b000000) #b1100000000)
       (or (bvsle (concat #x0000 k!45 k!44) #xffffffff)
           (bvsle #x00000001 (concat #x0000 k!45 k!44)))
       a!10
       (= k!28 #x34)
       (= k!29 #x00)
       (= k!30 #x00)
       (= k!31 #x00)
       (bvule #x0003 (concat k!45 k!44))
       a!11
       (bvule #x0004 (concat k!45 k!44))
       (bvule #x0005 (concat k!45 k!44))
       a!12
       (bvule #x0006 (concat k!45 k!44))
       (bvule #x0007 (concat k!45 k!44))
       a!13
       (bvule #x0008 (concat k!45 k!44))
       (bvule #x0009 (concat k!45 k!44))
       a!14
       a!16
       (bvule ((_ extract 3 0) k!44) #x9)
       a!17
       a!18
       a!19
       (= a!20 #x000000)
       (bvule a!21 #xdf)
       a!22
       (bvsle #x0000000000000000 (concat #x00000000 k!219 k!218 k!217 k!216))
       (bvsle (concat #x00000000 k!219 k!218 k!217 k!216) #x00000000000001cb)
       (or (bvsle (concat #x00000000 k!231 k!230 k!229 k!228)
                  #xffffffffffffffff)
           (bvsle #x0000000000000001
                  (concat #x00000000 k!231 k!230 k!229 k!228)))
       (or (bvsle (concat #x00000000 k!235 k!234 k!233 k!232)
                  #xffffffffffffffff)
           (bvsle #x0000000000000001
                  (concat #x00000000 k!235 k!234 k!233 k!232)))
       (= (bvnot a!25) #x00)
       a!24
       (bvsle #x0000000000000000 (concat #x00000000 k!231 k!230 k!229 k!228))
       (bvsle (concat #x00000000 k!231 k!230 k!229 k!228) #x7ffffffffffffffe)
       (bvule #x00000008 (concat k!231 k!230 k!229 k!228))
       (bvule (concat #x00000000 k!231 k!230 k!229 k!228) #xffffffffffffffbe)
       a!26
       a!27
       (bvule a!28 #x80)
       a!30
       (= a!31 #x0000000000000060)
       (= a!32 #x0000000000000000)
       (bvule #x00000005
              (bvshl #x00000001 (concat #b000000000000000000000000000 a!33)))
       (= ((_ extract 31 2) a!34) #b000000000000000000000000000000)
       (= (bvmul #x0000000000000004 (concat #x00000000 a!34))
          #x0000000000000004)
       (or (bvsle a!35 #x00000003) (bvsle #x00000005 a!35))
       (= (bvmul #x0000000000000004 (concat #x00000000 a!35))
          #x0000000000000008)
       (or a!36 a!37)
       (= k!228 #x64)
       (= k!229 #x00)
       (= k!230 #x00)
       (= k!231 #x00)
       (not a!38)
       (bvsle (concat #x00000000 k!231 k!230 k!229 k!228) #x0000000000800000)
       a!39
       (bvule #x00000020 (concat k!231 k!230 k!229 k!228))
       (= ((_ extract 2 0) k!216) #b000)
       (= ((_ extract 7 3) k!229) #b00000)
       (bvule (concat ((_ extract 2 0) k!229) k!228) #b10000000000)
       a!40
       (= k!216 #x68)
       (= k!217 #x01)
       (= k!218 #x00)
       (= k!219 #x00)))))

(maximize k!228)
(check-sat)