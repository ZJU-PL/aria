//@Description Binary tree

var {
    tree_node tree;
    int val;
    boolean elem_out;
}

relation {
    elem(tree, val, elem_out);
}

generator {
    boolean AP -> elem_out | !elem_out
                | is_empty(T) | !is_empty(T) 
                | compare(S, S + ??(1))
                | forall((x) -> compare(x, I), T)
                | exists((x) -> compare(x, I), T);
    int I -> val;
    int S -> tree_size(T) | 0;
    tree_node T -> tree;
}

example {
    int -> ??(2) | -1 * ??(2) ;
    boolean -> ??;
    tree_node(4) -> empty() | branch(int, tree_node, tree_node);
}