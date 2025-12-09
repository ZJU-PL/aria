//@Description 

var {
    list l;
    int n;
    list lout;
}

relation {
    take(l, n, lout);
}

generator {
    boolean AP -> compare(S, S + ??(1)) 
                | is_empty(L) | !is_empty(L) 
                | equal_list(L, L) | !equal_list(L, L);
    int S -> len(L) | n | 0 ;
    list L -> l | lout ;
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

void take(list l, int n, ref list ret) {
    assert (n >= 0);
    if (n > 0) {
        assert (l != null);
        
        ret = new list();
        ret.hd = l.hd;
        take(l.tl, n-1, ret.tl);
    } else {
        ret = null;        
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