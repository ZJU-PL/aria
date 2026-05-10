module BrokenLemmas

open FStar.List.Tot

// ─── Lemma 1: append_length ──────────────────────────────────────────
val append_length : l1:list 'a -> l2:list 'a ->
  Lemma (length (l1 @ l2) == length l1 + length l2)

let rec append_length l1 l2 =
  match l1 with
  | [] -> ()

// ─── Lemma 2: rev_append ─────────────────────────────────────────────
val rev_append : l1:list 'a -> l2:list 'a ->
  Lemma (rev (l1 @ l2) == rev l2 @ rev l1)

let rec rev_append l1 l2 =
  match l1 with
  | [] ->
    append_l_nil (rev l2)
  | hd :: tl ->
    rev_append l1 l2

// ─── Lemma 3: map_compose ────────────────────────────────────────────
val map_compose : #a:Type -> #b:Type -> #c:Type ->
  f:(a -> b) -> g:(b -> c) -> l:list a ->
  Lemma (map g (map f l) == map (fun x -> g (f x)) l)

let map_compose #a #b #c f g l =
  ()

// ─── Lemma 4: mem_append ─────────────────────────────────────────────
val mem_append : #a:eqtype -> x:a -> l1:list a -> l2:list a ->
  Lemma (mem x (l1 @ l2) == (mem x l1 || mem x l2))

let rec mem_append #a x l1 l2 =
  match l1 with
  | [] -> ()
  | hd :: tl ->
    mem_append x l2 tl

// ─── Lemma 5: length_filter_le ───────────────────────────────────────
val length_filter_le : #a:Type -> f:(a -> bool) -> l:list a ->
  Lemma (length (filter f l) <= length l)

let length_filter_le #a f l =
  admit ()
