//@Description 

var {
    list l1;
    list lout1;
    
    list l2;
    list lout2;

    int eps;
}

relation {
    stutter(l1, lout1);
    stutter(l2, lout2);
}

generator {
    boolean AP -> D1 > eps || D2 <= I;
    int D1 -> edist_dp(l1, l2);
    int D2 -> edist_dp(lout1, lout2);
    int CF -> ??(2) - 1;
    int I -> CF * len(l1) + CF * len(l2) + CF * eps + C;
    int C -> ??(2) - 1;
}

// To-Do: make sure only the constant hole (??) appears at most once
// ex. len(L) +- ?? | eps +- ??

example {
    int -> ??(4) | -1 * ??(4) ;
    list -> nil() | cons(int, list);
}