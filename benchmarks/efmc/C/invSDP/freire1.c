int main(){
  // variable declarations
  int x,r,a;
  //precondition
  assume(x>=0);
  r = 0;
  a = 2*x;
  // loop body
  while(r<=x){
    x = x - r;
    r = r + 1;
  }
  // post-condition
  assert(  (r*r + r >= a) && (r*r - r < a) );
}