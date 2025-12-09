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

void len(list l, ref int ret) {
    if (l == null) {
        ret = 0;
    } else {
        len(l.tl, ret);
        ret = ret + 1;
    }
}

void mem(list l, int x, ref boolean ret) {
    if (l == null) {
        ret = false;
    } else if (x == l.hd) {
        ret = true;
    } else {
        mem(l.tl, x, ret);
    }
}

void ord(list l, int x, int y, ref boolean ret) {
    if (l == null) {
        ret = false;
    } else if (x == l.hd) {
        mem(l.tl, y, ret);
    } else {
        ord(l.tl, x, y, ret);
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

void hdist(list l1, list l2, ref int ret) {
    if (l1 == null || l2 == null) {
        ret = 0;
    } else {
        hdist(l1.tl, l2.tl, ret);
        if (l1.hd != l2.hd) { ret = ret + 1; }
    }
}

void min3(int x, int y, int z, ref int ret) {
    ret = x;
    if (y < ret) { ret = y; }
    if (z < ret) { ret = z; }
}

void edist_dp(list l1, list l2, ref int ret) {
    int len1, len2;
    int r, c;

    len(l1, len1);
    len(l2, len2);

    r = len1 + 1;
    c = len2 + 1;

    int[2 * c] dp;

    int iter = 0;
    while(iter < c) {
        dp[iter] = iter;
        iter = iter + 1;
    }

    int flip = 1;
    int i = 1;

    int hd1, hd2;
    list l1_curr = l1;
    while (i < r) {
        hd1 = l1_curr.hd;

        dp[flip * c] = i;

        int j = 1;
        list l2_curr = l2;
        while (j < c) {
            hd2 = l2_curr.hd;

            if (hd1 == hd2) {
                dp[flip * c + j] = dp[(1 - flip) * c + (j - 1)];
            } else {
                min3(dp[flip * c + (j - 1)], dp[(1 - flip) * c + j], dp[(1 - flip) * c + (j - 1)], dp[flip * c + j]);
                dp[flip * c + j] = dp[flip * c + j] + 1;
            }

            l2_curr = l2_curr.tl;
            j = j + 1;
        }

        l1_curr = l1_curr.tl;
        flip = 1 - flip;
        i = i + 1;
    }

    ret = dp[(1 - flip) * c + len2];
}

void edist(list l1, list l2, ref int ret) {
    if (l1 == null) {
        len(l2, ret);
    } else if (l2 == null) {
        len(l1, ret);
    } else if (l1.hd == l2.hd) {
        edist(l1.tl, l2.tl, ret);
    } else {
        int insert_dist, remove_dist, replace_dist;

        edist(l1, l2.tl, insert_dist);
        edist(l1.tl, l2, remove_dist);
        edist(l1.tl, l2.tl, replace_dist);
        
        min3(insert_dist, remove_dist, replace_dist, ret);
        ret = ret + 1;
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

void delete(list l, int val, ref list ret) {
    if (l == null) {
        ret = null;
    } else if (l.hd == val) {
        ret = l.tl;
    } else {
        ret = new list();
        ret.hd = l.hd;
        delete(l.tl, val, ret.tl);
    }
}

void deleteAll(list l, int val, ref list ret) {
    if (l == null) {
        ret = null;
    } else if (l.hd == val) {
        deleteAll(l.tl, val, ret);
    } else {
        ret = new list();
        ret.hd = l.hd;
        deleteAll(l.tl, val, ret.tl);
    }
}

void drop(list l, int n, ref list ret) {
    assert (n >= 0);
    if (n > 0) {
        assert (l != null);
        drop(l.tl, n-1, ret);
    } else {
        ret = l;        
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

void stutter(list l, ref list ret) {
    if (l == null) {
        ret = null;    
    } else {
        list n1 = new list();
        list n2 = new list();

        n1.hd = l.hd;
        n2.hd = l.hd;

        n1.tl = n2;
        stutter(l.tl, n2.tl);
        ret = n1;
    }
}