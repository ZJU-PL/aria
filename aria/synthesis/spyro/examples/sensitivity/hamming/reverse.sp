//@Description 

var {
    list l1;
    list lout1;
    
    list l2;
    list lout2;

    int eps;
}

relation {
    reverse(l1, lout1);
    reverse(l2, lout2);
}

generator {
    boolean AP -> len(l1) != len(l2) || D1 > eps || D2 <= I;
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