//@Description 

var {
    int n;
    int val;
    list lout;
}

relation {
    replicate(n, val, lout);
}

generator {
    boolean AP -> is_empty(L) | !is_empty(L) 
                | compare(S, S + ??(1))
                | forall((x) -> compare(x, I), L) 
                | exists((x) -> compare(x, I), L);
    int I -> val ;
    int S -> len(L) | n | 0 ;
    list L -> lout ;
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

void replicate(int n, int val, ref list ret) {
    assert (n >= 0);
    if (n > 0) {
        ret = new list();

        ret.hd = val;
        replicate(n-1, val, ret.tl);
    } else {
        ret = null;        
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