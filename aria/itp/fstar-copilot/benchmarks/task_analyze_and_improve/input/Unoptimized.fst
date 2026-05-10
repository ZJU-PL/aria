module Unoptimized
#lang-pulse
open Pulse.Lib.Pervasives
open Pulse.Lib.Reference
open Pulse.Lib.Array
module A = Pulse.Lib.Array
module R = Pulse.Lib.Reference
module SZ = FStar.SizeT

let hash (key: int) (cap: nat{cap > 0}) : nat = key % cap

let rec mem_spec (x: int) (s: list int) : bool =
  match s with
  | [] -> false
  | hd :: tl -> hd = x || mem_spec x tl

let rec remove_spec (x: int) (s: list int) : list int =
  match s with
  | [] -> []
  | hd :: tl -> if hd = x then tl else hd :: remove_spec x tl

let insert_spec (x: int) (s: list int) : list int =
  if mem_spec x s then s else x :: s

let empty_slot : int = -1

let hashset_inv (data: A.array int) (sz: ref SZ.t) (cap: SZ.t)
  (s: erased (Seq.seq int)) (vsz: erased SZ.t) =
  A.pts_to data s **
  R.pts_to sz vsz **
  pure (Seq.length s == SZ.v cap /\ SZ.v cap > 0 /\ SZ.v vsz <= SZ.v cap)

#push-options "--z3rlimit 100"
fn hashset_create (cap: SZ.t{SZ.v cap > 0})
requires emp
returns r: (A.array int & ref SZ.t)
ensures exists* s vsz.
  hashset_inv (fst r) (snd r) cap s vsz
{
  let arr = A.alloc empty_slot cap;
  let sz = R.alloc 0sz;
  let r = (arr, sz);
  fold (hashset_inv (fst r) (snd r) cap (Seq.create (SZ.v cap) empty_slot) 0sz);
  r
}
#pop-options

#push-options "--z3rlimit 150"
fn hashset_insert (data: A.array int) (sz: ref SZ.t) (cap: SZ.t) (key: int)
  (#s: erased (Seq.seq int)) (#vsz: erased SZ.t)
requires hashset_inv data sz cap s vsz **
  pure (key <> empty_slot /\ SZ.v vsz < SZ.v cap)
ensures exists* s' vsz'.
  hashset_inv data sz cap s' vsz'
{
  unfold (hashset_inv data sz cap s vsz);
  let c = SZ.v cap in
  let mut i = SZ.uint_to_t (hash key (SZ.v cap));
  let mut found = false;
  while (
    let f = !found;
    not f
  )
  invariant exists* vi vfound s_cur vsz_cur.
    R.pts_to i vi **
    R.pts_to found vfound **
    A.pts_to data s_cur **
    R.pts_to sz vsz_cur **
    pure (
      SZ.v vi < SZ.v cap /\
      Seq.length s_cur == SZ.v cap /\
      SZ.v vsz_cur <= SZ.v cap
    )
  {
    let vi = !i;
    let slot = data.(vi);
    if slot = empty_slot then {
      data.(vi) <- key;
      let cur_sz = !sz;
      sz := SZ.add cur_sz 1sz;
      found := true
    } else {
      if slot = key then {
        found := true
      } else {
        let next = SZ.add vi 1sz;
        if SZ.lt next cap then
          i := next
        else
          i := 0sz
      }
    }
  };
  fold (hashset_inv data sz cap _ _)
}
#pop-options

#push-options "--z3rlimit 200"
fn hashset_contains (data: A.array int) (sz: ref SZ.t) (cap: SZ.t) (key: int)
  (#s: erased (Seq.seq int)) (#vsz: erased SZ.t)
requires hashset_inv data sz cap s vsz ** pure (key <> empty_slot)
returns b: bool
ensures hashset_inv data sz cap s vsz
{
  unfold (hashset_inv data sz cap s vsz);
  let mut i = SZ.uint_to_t (hash key (SZ.v cap));
  let mut result = false;
  let mut done = false;
  let mut steps = 0sz;
  while (
    let d = !done;
    not d
  )
  invariant exists* vi vr vd vs.
    R.pts_to i vi **
    R.pts_to result vr **
    R.pts_to done vd **
    R.pts_to steps vs **
    A.pts_to data #1.0R s **
    R.pts_to sz vsz **
    pure (SZ.v vi < SZ.v cap /\ Seq.length s == SZ.v cap /\ SZ.v vs <= SZ.v cap)
  {
    let vi = !i;
    let vs = !steps;
    if SZ.gte vs cap then {
      done := true
    } else {
      let slot = data.(vi);
      if slot = empty_slot then {
        done := true
      } else if slot = key then {
        result := true;
        done := true
      } else {
        let next = SZ.add vi 1sz;
        if SZ.lt next cap then
          i := next
        else
          i := 0sz;
        steps := SZ.add vs 1sz
      }
    }
  };
  let r = !result;
  fold (hashset_inv data sz cap s vsz);
  admit();
  r
}
#pop-options

fn hashset_size (data: A.array int) (sz: ref SZ.t) (cap: SZ.t)
  (#s: erased (Seq.seq int)) (#vsz: erased SZ.t)
requires hashset_inv data sz cap s vsz
returns r: SZ.t
ensures hashset_inv data sz cap s vsz ** pure (r == reveal vsz)
{
  unfold (hashset_inv data sz cap s vsz);
  let r = !sz;
  admit();
  r
}
