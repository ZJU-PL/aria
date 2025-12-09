//@Description Binary tree

var {
    int val;
    tree_node t_left;
    tree_node t_right;
    tree_node t_out;

    tree_node rootval_in;
    int rootval_out;
}

relation {
    branch(val, t_left, t_right, t_out);
    root_val(rootval_in, rootval_out);
}

generator {
    boolean AP -> compare(I, I)
                | is_empty(T) | !is_empty(T)
                | tree_equal(T, T) | !tree_equal(T, T);
    int I -> val | rootval_out ;
    tree_node T -> t_left | t_right | t_out | rootval_in ;
}

example {
    int -> ??(2) | -1 * ??(2) ;
    tree_node(4) -> empty() | branch(int, tree_node, tree_node);
}