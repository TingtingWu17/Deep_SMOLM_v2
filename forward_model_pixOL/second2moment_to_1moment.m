syms Mxx Myy Mzz Mxy Mxz Myz

M == [Mxx, Mxy, Mxz;
      Mxy, Myy, Myz;
      Mxz, Myz, Mzz];
  
  
D = eig(M)
