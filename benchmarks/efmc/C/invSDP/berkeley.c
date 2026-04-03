int main(){
  // variable declarations
  int u,e,n,i,t;
  //precondition
  assume(u==0);
  assume(e==0);
  assume(n==0);
  assume(i>=1);
  assume(t==i);
  // loop body
  while(unknown()){
        if (i>=1){
            u = u + 1;
            e = 0;
            n = n + e;
            i = i - 1;
        } 
        else if (i>=1)  
        {
            u = 0;
            e = 1;
            n = 0;
            i = i + e + n + u - 1;    
        } 
        else if (n>=1)
        {
            u = 0;
            e = e + 1;
            n = 0;
            i = i + n + u - 1;        
        }
        else
        {
            u = 0;
            e = e + 1;
            n = 0;
            i = i + n + u - 1;     
        }
  }
  // post-condition
  assert( t == u + e + n + i);
}