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
    cons(x1, l1, lout1);
    cons(x2, l2, lout2);
}

generator {
    boolean AP -> !G || D1 > eps || D2 <= I;
    boolean G -> true | x1 == x2 | x1 != x2;
    int D1 -> edist_dp(l1, l2);
    int D2 -> edist_dp(lout1, lout2);
    int CF -> ??(2) - 1;
    int I -> CF * len(l1) + CF * len(l2) + CF * eps + CF;
}

// To-Do: make sure only the constant hole (??) appears at most once
// ex. len(L) +- ?? | eps +- ??

example {
    int -> ??(4) | -1 * ??(4) ;
    list -> nil() | cons(int, list);
}