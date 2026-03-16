int main(){
  // variable declarations
  int x,s,r,a;
  //precondition
  assume(x==a);
  assume(r==1);
  assume(s==3.25); // not applicable
  assume(a>=0);

  // loop body
  while(s<=x){
    x = x - s;
    r = r + 1;
    s = s + 6*r + 3;
  }
  // post-condition
  assert(  (4*a-4*r*r*r-6*r*r-3*r<=0) && (4*r*r*r-6*r*r+3*r-1-4*a<=0) );
}