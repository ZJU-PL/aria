(set-logic BV)

(synth-inv inv_fun ((glob2_pathlim_off (_ BitVec 64))(glob2_pathbuf_off (_ BitVec 64))(glob2_p_off (_ BitVec 64))(bound_off (_ BitVec 64))(pathbuf_off (_ BitVec 64))(MAXPATHLEN (_ BitVec 64))))

(define-fun pre_fun ((glob2_pathlim_off (_ BitVec 64)) (glob2_pathbuf_off (_ BitVec 64)) (glob2_p_off (_ BitVec 64)) (bound_off (_ BitVec 64)) (pathbuf_off (_ BitVec 64)) (MAXPATHLEN (_ BitVec 64))) Bool
       ( and ( bvugt MAXPATHLEN #x0000000000000000 ) ( bvult MAXPATHLEN #x000000007fffffff ) ( = pathbuf_off #x0000000000000000 ) ( = bound_off ( bvsub ( bvadd pathbuf_off MAXPATHLEN #x0000000000000001 ) #x0000000000000001 ) ) ( = glob2_pathbuf_off pathbuf_off ) ( = glob2_pathlim_off bound_off ) ( = glob2_p_off glob2_pathbuf_off ) ))
(define-fun trans_fun ((glob2_pathlim_off! (_ BitVec 64)) (glob2_pathbuf_off! (_ BitVec 64)) (glob2_p_off! (_ BitVec 64)) (bound_off! (_ BitVec 64)) (pathbuf_off! (_ BitVec 64)) (MAXPATHLEN! (_ BitVec 64)) (glob2_pathlim_off (_ BitVec 64)) (glob2_pathbuf_off (_ BitVec 64)) (glob2_p_off (_ BitVec 64)) (bound_off (_ BitVec 64)) (pathbuf_off (_ BitVec 64)) (MAXPATHLEN (_ BitVec 64))) Bool
       ( and ( bvule glob2_p_off glob2_pathlim_off ) ( = glob2_p_off ( bvadd glob2_p_off #x0000000000000001 ) ) ( = MAXPATHLEN! MAXPATHLEN ) ( = pathbuf_off! pathbuf_off ) ( = bound_off! bound_off ) ( = glob2_pathbuf_off! glob2_pathbuf_off ) ( = glob2_pathlim_off! glob2_pathlim_off ) ))
(define-fun post_fun ((glob2_pathlim_off (_ BitVec 64)) (glob2_pathbuf_off (_ BitVec 64)) (glob2_p_off (_ BitVec 64)) (bound_off (_ BitVec 64)) (pathbuf_off (_ BitVec 64)) (MAXPATHLEN (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( = glob2_p_off #x0000000000000000 ) ( not ( bvule ( bvadd #x0000000000000001 MAXPATHLEN ) glob2_p_off ) ) ) ) ) ( or ( not ( bvule glob2_p_off glob2_pathlim_off ) ) a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

