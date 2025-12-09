// @Description Synthesize the property of list reverse function.
// Run with --inline-bnd-amnt 10 --num-atom-max 3

// Input and output variables
var {
    list l;
    list lout;
}

// Target functions that we aim to synthesize specifications
// The last argument to the function is output
relation {
    reverse(l, lout);
}

// The DSL for the specifictions.
// It uses the top predicate as a grammar for each disjunct.
// The maximum number of disjuncts are provided by the option "--num-atom-max"
// compare is macro for { == | != | <= | >= | < | > }
// ??(n) denotes arbitrary positive integer of n bits
// Provide only input arguments to function call
generator {
    boolean AP -> is_empty(L) | !is_empty(L) 
                | equal_list(L, L) | !equal_list(L, L)
                | compare(S, S + ??(1));            
    int S -> len(L) | 0 ;
    list L -> l | lout ;
}

// recursive constructor for each type
// Provide only input arguments to function call
// integer is chosen from arbitrary positive or negative 3-bits integer
example {
    int -> ??(3) | -1 * ??(3) ;
    list -> nil() | cons(int, list);
}

struct list {
    int hd;
	list tl;
}

// Implementation of all the functions

// The return value of function is passed by reference
void nil(ref list ret) {
    ret = null;
}

void cons(int hd, list tl, ref list ret) {
    ret = new list();
    ret.hd = hd;
    ret.tl = tl;
}

// Only consider the case when l is not null (empty)
void head(list l, ref int ret) {
    assert (l != null);

    ret = l.hd;
}

// Only consider the case when l is not null (empty)
void tail(list l, ref list ret) {
    assert (l != null);

    ret = l.tl;
}

void list_copy(list l, ref list ret) {
    if (l == null) {
        ret = null;
    } else {
        ret = new list();
        ret.hd = l.hd;

        list tl_copy;
        list_copy(l.tl, tl_copy);
        ret.tl = tl_copy;
    } 
}

void snoc(list l, int val, ref list ret) {
    if (l == null) {
        ret = new list();
        ret.hd = val;
        ret.tl = null;
    } else {
        ret = new list();
        ret.hd = l.hd;
        snoc(l.tl, val, ret.tl);
    }
}

void reverse(list l, ref list ret) {
    if (l == null) {
        ret = null;
    } else {
        list tl_reverse;
        reverse(l.tl, tl_reverse);
        snoc(tl_reverse, l.hd, ret);
    }
}

void len(list l, ref int ret) {
    if (l == null) {
        ret = 0;
    } else {
        len(l.tl, ret);
        ret = ret + 1;
    }
}

void is_empty(list l, ref boolean ret) {
    ret = (l == null);
}

void equal_list(list l1, list l2, ref boolean ret) {
    if (l1 == null || l2 == null) {
        ret = l1 == l2;
    } else {
        equal_list(l1.tl, l2.tl, ret);
        ret = l1.hd == l2.hd && ret;
    }
}