# Matrix-Factorization
This repository provides MATLAB implementations for various Matrix Factorization methods. The matrix factorization methods available in this repository are:

## Single Matrix Factorization Methods

### ONMF
  - X = W*H
  - columns of W are orthogonal
  - values of W and H are non-negative
  
Ding, Chris, et al. "Orthogonal nonnegative matrix t-factorizations for clustering." Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining. 2006.

### OMUR
  - X = W*H
  - W is projected onto the Stiefel manifold
  - values of W and H are non-negative
  
Choi, Seungjin. "Algorithms for orthogonal nonnegative matrix factorization." 2008 ieee international joint conference on neural networks (ieee world congress on computational intelligence). IEEE, 2008.

### OAUR
  - X = W*H
  - orthogonality penalty on the columns of W
  - values of W and H are non-negative

Mirzal, Andri. "A convergent algorithm for orthogonal nonnegative matrix factorization." Journal of Computational and Applied Mathematics 260 (2014): 149-166.

### OHALS
  - X = W*H
  - columns of W are orthogonal
  - values of W and H are non-negative
  - hierarchical ALS solution

Kimura, Keigo, Yuzuru Tanaka, and Mineichi Kudo. "A fast hierarchical alternating least squares algorithm for orthogonal nonnegative matrix factorization." Asian Conference on Machine Learning. 2015.

### LVNMF
  - X = W*H
  - columns of W are maximally different
  - values of W and H are non-negative
  
Liu, Tongliang, Mingming Gong, and Dacheng Tao. "Large-cone nonnegative matrix factorization." IEEE transactions on neural networks and learning systems 28.9 (2016): 2129-2142.

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

You must convert X into a non-negative format (either through adding an offset or taking the absolute value) to use ONMTF_SCR.

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