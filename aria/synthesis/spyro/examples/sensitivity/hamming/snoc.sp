//@Description 

var {
    list l1;
    int x1;
    list lout1;
    
    list l2;
    int x2;
    list lout2;

    int eps;
}

relation {
    snoc(l1, x1, lout1);
    snoc(l2, x2, lout2);
}

generator {
    boolean AP -> !G || len(l1) != len(l2) || D1 > eps || D2 <= I;
    boolean G -> true | x1 == x2 | x1 != x2;
    int D1 -> hdist(l1, l2);
    int D2 -> hdist(lout1, lout2);
    int CF -> ??(2) - 1;
    int I -> CF * len(l1) + CF * eps + CF;
}

// To-Do: make sure only the constant hole (??) appears at most once
// ex. len(L) +- ?? | eps +- ??

example {
    int -> ??(4) | -1 * ??(4) ;
    list -> nil() | cons(int, list);
}