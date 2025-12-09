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
    boolean AP -> len(l11) != len(l21) || len(l12) != len(l22) || D1 > eps1 || D2 > eps2 || D3 <= I;
    int D1 -> hdist(l11, l21);
    int D2 -> hdist(l12, l22);
    int D3 -> hdist(lout1, lout2);
    int CF -> ??(2) - 1;
    int IL -> CF * len(l11) + CF * len(l12);
    int I -> IL + CF * eps1 + CF * eps2 + CF;
}

// To-Do: make sure only the constant hole (??) appears at most once
// ex. len(L) +- ?? | eps +- ??

example {
    int -> ??(4) | -1 * ??(4) ;
    list -> nil() | cons(int, list);
}