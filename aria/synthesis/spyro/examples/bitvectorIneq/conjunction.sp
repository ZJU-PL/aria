var {
    int x;
    int y;
    boolean b;
}

relation {
    rel(x, y, b);
}

generator {
    boolean INEQ -> !b || bvadd3(CX, CY, C) <= bvadd3(CX, CY, C);
    int CX -> bvmul(C, x);
    int CY -> bvmul(C, y);
    int C -> ??(4);
}

example {
    int -> ??(4);
    boolean -> ??;
}

void bvadd(int x, int y, ref int ret) {
    ret = (x + y) % 16; 
}

void bvadd3(int x, int y, int z, ref int ret) {
    ret = (x + y + z) % 16; 
}

void bvmul(int x, int y, ref int ret) {
    ret = (x * y) % 16;
}

void rel(int x, int y, ref boolean ret) {
    ret = ((x + y + 4) % 16 <= (x + 15 * y + 7) % 16);
    ret = ret && ((2 * x + 14 * y + 3) % 16 <= (x + 15 * y + 7) % 16);
}