//@Description Binary tree

var {
    int val;
    tree_node t_left;
    tree_node t_right;
    tree_node t_out;
}

relation {
    branch(val, t_left, t_right, t_out);
}

generator {
    boolean AP -> tree_equal(T, T) | !tree_equal(T, T)
                | is_empty(T) | !is_empty(T) 
                | compare(S, S + S + ??(1))
                | forall((x) -> compare(x, I), T)
                | exists((x) -> compare(x, I), T);
    int I -> val;
    int S -> tree_size(T) | 0;
    tree_node T -> t_left | t_right | t_out;
}

example {
    int -> ??(2) | -1 * ??(2) ;
    tree_node(4) -> empty() | branch(int, tree_node, tree_node);
}