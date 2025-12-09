//@Description 

var {
    list l;
    int min_out;
}

relation {
    min(l, min_out);
}

generator {
    boolean AP -> compare(S, S + ??(1))
                | is_empty(L) | !is_empty(L)
                | forall((x) -> compare(x, I), L) 
                | exists((x) -> compare(x, I), L);
    int S -> len(L) | 0 ;
    list L -> l ;
    int I -> min_out ;
}

example {
    int -> ??(3) | -1 * ??(3) ;
    list -> nil() | cons(int, list);
}

struct list {
    int hd;
	list tl;
}

void nil(ref list ret) {
    ret = null;
}

void cons(int hd, list tl, ref list ret) {
    ret = new list();
    ret.hd = hd;
    ret.tl = tl;
}

void head(list l, ref int ret) {
    assert (l != null);

    ret = l.hd;
}

void tail(list l, ref list ret) {
    assert (l != null);

    ret = l.tl;
}

void min(list l, ref int ret) {
    assert (l != null);

    if (l.tl != null) {
        min(l.tl, ret);
        if (l.hd < ret) {
            ret = l.hd;
        }
    } else {
        ret = l.hd;
    }
}

void forall(fun f, list l, ref boolean ret) {
    if (l == null) {
        ret = true;
    } else {
        forall(f, l.tl, ret);        
        ret = ret && f(l.hd);
    }
}

void exists(fun f, list l, ref boolean ret) {
    if (l == null) {
        ret = false;
    } else {
        exists(f, l.tl, ret);
        ret = ret || f(l.hd);
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