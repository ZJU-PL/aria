int main(){
  // variable declarations
  int n,a,su,t;
  //precondition
  assume(a==0);
  assume(su==1);
  assume(t==1);
  assume(n>=0);
  // loop body
  while(su<=n){
    a = a + 1;
    su = su + t + s;
    t = t + 2;
  }
  // post-condition
  assert(  (a*a<=n) && (n<=(a+1)*(a+1)) );
}