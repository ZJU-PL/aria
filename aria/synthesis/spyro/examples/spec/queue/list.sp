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

void append(list l1, list l2, ref list ret) {
    if (l1 == null) {
        ret = l2;
    } else {
        list tl_append;
        append(l1.tl, l2, tl_append);
        cons(l1.hd, tl_append, ret);
    }
}

void list_len(list l, ref int ret) {
    if (l == null) {
        ret = 0;
    } else {
        list_len(l.tl, ret);
        ret = ret + 1;
    }
}

void is_empty_list(list l, ref boolean ret) {
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