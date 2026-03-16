int main(){
  // variable declarations
  int m,s,e,i,t;
  //precondition
  assume(m==0);
  assume(s==0);
  assume(e==0);
  assume(t==i);
  assume(i>=1);
  // loop body
  while(unknown()){
        if (vi>=1){
            m = 0;
            s = s + e + m + 1;
            e = 0;
            i = i - 1;
        } 
        else if (ve>=1)
        {
            m = m + 1;
            s = 0;
            e = e - 1;
        }        
        else if (vs>=1)
        {
            m = 0;
            s = 0;
            e = 1;
            i = i + m + e + s - 1;
        }
        else
        {
            m = 0;
            s = 0;
            e = 1;
            i = i + m + e + s - 1;
        }
    }
  // post-condition
  assert( t == m + s + e + i);
}