# Matrix-Factorization
This repository provides MATLAB implementations for various Matrix Factorization methods. The matrix factorization methods available in this repository are:

## Single Matrix Factorization Methods

## Single Matrix Tri-Factorization Methods

### Fast MtF
  - X(i) = F(i)*S(i)*F(i)'
  - Each row in F(i) contains one non-zero value and the values in F(i) are constrained to be one or zero.
  - S(i) = inv(F(i)'*F(i))*F(i)'*X*F(i)*inv(F(i)'*F(i))

Wang, Hua, et al. "Fast nonnegative matrix tri-factorization for large-scale data co-clustering." Twenty-Second International Joint Conference on Artificial Intelligence. 2011.

### ONMTF_SCR
  - X(i) = F(i)*S(i)*F(i)'
  - F(i)'*F(i) = I
  - values of F(i) and S(i) are non-negative
  - clusters in F(i) are spatially cohesive

Bai, Zilong, et al. "Unsupervised network discovery for brain imaging data." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.

## Group Matrix Tri-Factorization Methods

### LMF
  - X(i) = F*S(i)*F'
  - no additional constraints or penalties

Tang, Wei, Zhengdong Lu, and Inderjit S. Dhillon. "Clustering with multiple graphs." 2009 Ninth IEEE International Conference on Data Mining. IEEE, 2009.

### RESCAL
  - X(i) = F*S(i)*F'
  - no additional constraints or penalties

Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel. "A three-way model for collective learning on multi-relational data." Icml. Vol. 11. 2011.

### GLEAN
  - X(i) = F*S(i)*F'
  - F'*F = I
  - values of F are non-negative
  - S(i) = F'*X*F

Li, Kendrick T. Group Convex Orthogonal Non-negative Matrix Tri-Factorization with Applications in FC Fingerprinting. Diss. University of Cincinnati, 2020.

# Additional Information

## Matrix Tri-Factorization Details
Use core functions (fastMtF, GLEAN, LMF, RESCAL, ONMTF_SCR) if you wish to run a single factorization on X (a set of symmetric 2D matrices). Use multi functions (multiFastMtF, multiGLEAN, multiLMF, multiRESCAL, multiONMTF_SCR) if you wish to run multiple runs using a random initialization. These functions will return the run with the smallest error and the results for all runs.

## Single Matrix Factorization Details
You must convert X into a non-negative format (either through adding an offset or taking the absolute value) to use ONMTF_SCR.