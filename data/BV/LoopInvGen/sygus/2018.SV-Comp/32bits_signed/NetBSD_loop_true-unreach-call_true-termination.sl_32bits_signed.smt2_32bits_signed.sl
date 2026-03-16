(set-logic BV)

(synth-inv inv_fun ((glob2_pathlim_off (_ BitVec 32))(glob2_pathbuf_off (_ BitVec 32))(glob2_p_off (_ BitVec 32))(bound_off (_ BitVec 32))(pathbuf_off (_ BitVec 32))(MAXPATHLEN (_ BitVec 32))))

(define-fun pre_fun ((glob2_pathlim_off (_ BitVec 32)) (glob2_pathbuf_off (_ BitVec 32)) (glob2_p_off (_ BitVec 32)) (bound_off (_ BitVec 32)) (pathbuf_off (_ BitVec 32)) (MAXPATHLEN (_ BitVec 32))) Bool
       ( and ( bvsgt MAXPATHLEN #x00000000 ) ( bvslt MAXPATHLEN #x7fffffff ) ( = pathbuf_off #x00000000 ) ( = bound_off ( bvsub ( bvadd pathbuf_off MAXPATHLEN #x00000001 ) #x00000001 ) ) ( = glob2_pathbuf_off pathbuf_off ) ( = glob2_pathlim_off bound_off ) ( = glob2_p_off glob2_pathbuf_off ) ))
(define-fun trans_fun ((glob2_pathlim_off! (_ BitVec 32)) (glob2_pathbuf_off! (_ BitVec 32)) (glob2_p_off! (_ BitVec 32)) (bound_off! (_ BitVec 32)) (pathbuf_off! (_ BitVec 32)) (MAXPATHLEN! (_ BitVec 32)) (glob2_pathlim_off (_ BitVec 32)) (glob2_pathbuf_off (_ BitVec 32)) (glob2_p_off (_ BitVec 32)) (bound_off (_ BitVec 32)) (pathbuf_off (_ BitVec 32)) (MAXPATHLEN (_ BitVec 32))) Bool
       ( and ( bvsle glob2_p_off glob2_pathlim_off ) ( = glob2_p_off ( bvadd glob2_p_off #x00000001 ) ) ( = MAXPATHLEN! MAXPATHLEN ) ( = pathbuf_off! pathbuf_off ) ( = bound_off! bound_off ) ( = glob2_pathbuf_off! glob2_pathbuf_off ) ( = glob2_pathlim_off! glob2_pathlim_off ) ))
(define-fun post_fun ((glob2_pathlim_off (_ BitVec 32)) (glob2_pathbuf_off (_ BitVec 32)) (glob2_p_off (_ BitVec 32)) (bound_off (_ BitVec 32)) (pathbuf_off (_ BitVec 32)) (MAXPATHLEN (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( bvsle glob2_p_off #x00000000 ) ( not ( bvsle ( bvadd #x00000001 MAXPATHLEN ) glob2_p_off ) ) ) ) ) ( or ( not ( bvsle glob2_p_off glob2_pathlim_off ) ) a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

