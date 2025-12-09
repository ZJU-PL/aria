//@Description 

var {
    list l11;
    list l12;
    list lout1;
    
    list l21;
    list l22;
    list lout2;

    int eps1;
    int eps2;
}

relation {
    append(l11, l12, lout1);
    append(l21, l22, lout2);
}

generator {
    boolean AP -> D1 > eps1 || D2 > eps2 || D3 <= I;
    int D1 -> edist_dp(l11, l21);
    int D2 -> edist_dp(l12, l22);
    int D3 -> edist_dp(lout1, lout2);
    int CF -> ??(2) - 1;
    int IL -> CF * len(l11) + CF * len(l12) + CF * len(l21) + CF * len(l22);
    int I -> IL + CF * eps1 + CF * eps2 + CF;
}

// To-Do: make sure only the constant hole (??) appears at most once
// ex. len(L) +- ?? | eps +- ??

example {
    int -> ??(4) | -1 * ??(4) ;
    list -> nil() | cons(int, list);
}