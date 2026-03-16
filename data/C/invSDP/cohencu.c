int main(){
  // variable declarations
  int n,x,y,z;
  //precondition
  assume(n==0);
  assume(x==0);
  assume(y==1);
  assume(z==6);
  // loop body
  while(unknown()){
       //assert(z == 6*n + 6);
       //assert(y == 3*n*n + 3*n + 1);
       //assert(x == n*n*n);
       //%%%traces: int n, int x, int y, int z
       n=n+1;
       x=x+y;
       y=y+z;
       z=z+6;
  }
  // post-condition
  assert( (z == 6*n + 6) && (y == 3*n*n + 3*n + 1) && (x == n*n*n));
}