module WeakSpec
open Pulse.Lib.Pervasives
open Pulse.Lib.Reference
module R = Pulse.Lib.Reference

noeq
type store : Type0

val store_inv : store -> erased int -> erased int -> erased bool -> slprop

val create (u: unit)
  : stt store emp (fun s -> exists* vk vv. store_inv s vk vv false)

val insert (s: store) (k v: int)
  (#vk #vv: erased int) (#vhas: erased bool)
  : stt unit
      (store_inv s vk vv vhas)
      (fun _ -> exists* vk' vv' vhas'. store_inv s vk' vv' vhas')

val lookup (s: store) (k: int)
  (#vk #vv: erased int) (#vhas: erased bool)
  : stt int
      (store_inv s vk vv vhas)
      (fun _ -> store_inv s vk vv vhas)

val delete (s: store) (k: int)
  (#vk #vv: erased int) (#vhas: erased bool)
  : stt unit
      (store_inv s vk vv vhas)
      (fun _ -> exists* vk' vv' vhas'. store_inv s vk' vv' vhas')
