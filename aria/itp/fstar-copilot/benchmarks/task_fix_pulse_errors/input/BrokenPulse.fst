module BrokenPulse
#lang-pulse
open Pulse.Lib.Pervasives
open Pulse.Lib.Reference
open Pulse.Lib.Array
module A = Pulse.Lib.Array
module R = Pulse.Lib.Reference
module SZ = FStar.SizeT

// ─── Bounded buffer predicate ────────────────────────────────────────

let buffer_inv (buf: A.array int) (head tail len: ref SZ.t) (cap: SZ.t)
  (s: erased (Seq.seq int)) (vh vt vl: erased SZ.t) =
  A.pts_to buf s **
  R.pts_to head vh **
  R.pts_to tail vt **
  R.pts_to len vl **
  pure (
    Seq.length s == SZ.v cap /\
    SZ.v vh < SZ.v cap /\
    SZ.v vt < SZ.v cap /\
    SZ.v vl <= SZ.v cap
  )

// ─── Function 1 ─────────────────────────────────────────────────────

fn is_empty (len: ref SZ.t) (#vl: erased SZ.t)
requires R.pts_to len vl
returns b: bool
ensures R.pts_to len vl ** pure (b == (SZ.v vl = 0))
{
  with vl_ghost. _;
  if (SZ.v vl_ghost = 0) {
    let _ = !len;
    true
  } else {
    false
  }
}

// ─── Function 2 ─────────────────────────────────────────────────────

fn read_length (buf: A.array int) (head tail len: ref SZ.t) (cap: SZ.t)
  (#s: erased (Seq.seq int)) (#vh #vt #vl: erased SZ.t)
requires buffer_inv buf head tail len cap s vh vt vl
returns r: SZ.t
ensures buffer_inv buf head tail len cap s vh vt vl ** pure (r == reveal vl)
{
  unfold (buffer_inv buf head tail len cap s vh vt vl);
  let r = !len;
  r
}

// ─── Function 3 ─────────────────────────────────────────────────────

fn clear_buffer (buf: A.array int) (len: ref SZ.t) (cap: SZ.t)
  (#s: erased (Seq.seq int)) (#vl: erased SZ.t)
requires A.pts_to buf s ** R.pts_to len vl ** pure (Seq.length s == SZ.v cap)
ensures exists* s' vl'.
  A.pts_to buf s' ** R.pts_to len vl' **
  pure (SZ.v vl' == 0 /\ Seq.length s' == SZ.v cap)
{
  let mut i = 0sz;
  while (
    let vi = !i;
    SZ.lt vi cap
  )
  invariant b. exists* vi s_cur.
    R.pts_to i vi **
    A.pts_to buf s_cur **
    R.pts_to len vl **
    pure (SZ.v vi <= SZ.v cap /\ Seq.length s_cur == SZ.v cap)
  {
    let vi = !i;
    buf.(vi) <- 0;
    i := SZ.add vi 1sz
  };
  len := 0sz
}

// ─── Function 4 ─────────────────────────────────────────────────────

fn advance_head (head: ref SZ.t) (cap: SZ.t)
  (#vh: erased SZ.t)
requires R.pts_to head vh ** pure (SZ.v vh < SZ.v cap)
ensures exists* vh'.
  R.pts_to head vh' **
  pure (SZ.v vh' == (SZ.v vh + 1) % SZ.v cap)
{
  let h = !head;
  let next = SZ.add h 1sz;
  if (SZ.lt next cap) {
    head := next
  } else {
    head := 0sz
  }
}

// ─── Function 5 ─────────────────────────────────────────────────────

fn read_at_head (buf: A.array int) (head: ref SZ.t) (cap: SZ.t)
  (#s: erased (Seq.seq int)) (#vh: erased SZ.t)
requires A.pts_to buf s ** R.pts_to head vh **
  pure (SZ.v vh < Seq.length s /\ Seq.length s == SZ.v cap)
returns r: int
ensures A.pts_to buf s ** R.pts_to head vh ** pure (r == Seq.index s (SZ.v vh))
{
  let h = !head;
  rewrite (R.pts_to head vh) as (A.pts_to buf s);
  let v = buf.(h);
  v
}
