//@Description Binary tree

var {
    tree_node bst;
    int target;
    boolean find_out;
}

relation {
    bst_find(bst, target, find_out);
}

generator {
    boolean AP -> compare(S, S + ??(1))
                | find_out | !find_out
                | tree_equal(T, T) | !tree_equal(T, T)
                | is_empty(T) | !is_empty(T)
                | forall((x) -> compare(x, I), T)
                | exists((x) -> compare(x, I), T);
    int I -> target;
    int S -> tree_size(T) | 0;
    tree_node T -> bst;
}

example {
    int -> ??(2) | -1 * ??(2) ;
    boolean -> ??;
    tree_node(4) -> bst_empty() | bst_insert(tree_node, int);
}

struct tree_node {
    int val;
	tree_node left;
    tree_node right;	
}

void bst_empty(ref tree_node ret) {
    ret = null;
}

void bst_insert(tree_node t, int val, ref tree_node ret) {
    if (t == null) {
        ret = new tree_node();
        ret.val = val;
        ret.left = null;
        ret.right = null;
    } else if (val < t.val) {
        ret = new tree_node();
        ret.val = t.val;
        bst_insert(t.left, val, ret.left);
        ret.right = t.right;
    } else if (val > t.val) {
        ret = new tree_node();
        ret.val = t.val;
        ret.left = t.left;
        bst_insert(t.right, val, ret.right);
    } else {
        // Duplicate case. Duplicates are excluded in this example.
        ret = t;
    }
}

void bst_minValue(tree_node t, ref int ret) {
    assert (t != null);

    if (t.left == null) {
        ret = t.val;
    } else {
        bst_minValue(t.left, ret);
    }
}

void bst_find(tree_node t, int target, ref boolean ret) {
    if (t == null){
        ret = false;
    } else if (target == t.val) {
        ret = true;
    } else if (target < t.val) {
        bst_find(t.left, target, ret);
    } else {
        bst_find(t.right, target, ret);
    }
}

void is_empty(tree_node t, ref boolean ret) {
    ret = (t == null);
}

void tree_size(tree_node t, ref int ret) {
    if (t == null) {
        ret = 0;
    } else {
        int size_left;
        int size_right;
        tree_size(t.left, size_left);
        tree_size(t.right, size_right);
        ret = size_left + size_right + 1;
    }
}

void tree_equal(tree_node t1, tree_node t2, ref boolean ret) {
    if (t1 == null || t2 == null) {
        ret = t1 == t2;
    } else {
        boolean left_equal;
        boolean right_equal;
        tree_equal(t1.left, t2.left, left_equal);
        tree_equal(t1.right, t2.right, right_equal);
        ret = t1.val == t2.val && left_equal && right_equal;
    }
}

void forall(fun f, tree_node t, ref boolean ret) {
    if (t == null) {
        ret = true;
    } else {
        boolean left_ret;
        boolean right_ret;

        forall(f, t.left, left_ret); 
        forall(f, t.right, right_ret);

        ret = left_ret && right_ret && f(t.val);
    }
}

void exists(fun f, tree_node t, ref boolean ret) {
    if (t == null) {
        ret = false;
    } else {
        boolean left_ret;
        boolean right_ret;

        exists(f, t.left, left_ret); 
        exists(f, t.right, right_ret);

        ret = left_ret || right_ret || f(t.val);
    }
}