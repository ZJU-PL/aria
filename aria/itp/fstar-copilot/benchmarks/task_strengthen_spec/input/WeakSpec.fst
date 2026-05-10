module WeakSpec
#lang-pulse
open Pulse.Lib.Pervasives
open Pulse.Lib.Reference
module R = Pulse.Lib.Reference

noeq
type store = {
  key_ref: ref int;
  val_ref: ref int;
  has_entry: ref bool;
}

let store_inv (s: store) (vk vv: erased int) (vhas: erased bool) =
  R.pts_to s.key_ref vk **
  R.pts_to s.val_ref vv **
  R.pts_to s.has_entry vhas

fn create (u: unit)
requires emp
returns s: store
ensures exists* vk vv. store_inv s vk vv false
{
  let kr = R.alloc 0;
  let vr = R.alloc 0;
  let hr = R.alloc false;
  let s = { key_ref = kr; val_ref = vr; has_entry = hr };
  fold (store_inv s 0 0 false);
  s
}

fn insert (s: store) (k v: int)
  (#vk #vv: erased int) (#vhas: erased bool)
requires store_inv s vk vv vhas
ensures exists* vk' vv' vhas'. store_inv s vk' vv' vhas'
{
  unfold (store_inv s vk vv vhas);
  s.key_ref := k;
  s.val_ref := v;
  s.has_entry := true;
  fold (store_inv s k v true)
}

fn lookup (s: store) (k: int)
  (#vk #vv: erased int) (#vhas: erased bool)
requires store_inv s vk vv vhas
returns r: int
ensures store_inv s vk vv vhas
{
  unfold (store_inv s vk vv vhas);
  let has = !s.has_entry;
  let sk = !s.key_ref;
  let sv = !s.val_ref;
  fold (store_inv s vk vv vhas);
  if has && sk = k then sv else 0
}

fn delete (s: store) (k: int)
  (#vk #vv: erased int) (#vhas: erased bool)
requires store_inv s vk vv vhas
ensures exists* vk' vv' vhas'. store_inv s vk' vv' vhas'
{
  unfold (store_inv s vk vv vhas);
  let has = !s.has_entry;
  let sk = !s.key_ref;
  if has && sk = k then {
    s.has_entry := false;
    fold (store_inv s vk vv false)
  } else {
    fold (store_inv s vk vv vhas)
  }
}
