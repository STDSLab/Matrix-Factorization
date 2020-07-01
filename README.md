# Matrix-Tri-Factorization

These functions implement the various matrix factorization methods we tested in our paper.

Use core functions (fastMtF, GLEAN, LMF, RESCAL, ONMTF_SCR) if you wish to run a single factorization on X (a set of symmetric 2D matrices).

Use multi functions (multiFastMtF, multiGLEAN, multiLMF, multiRESCAL, multiONMTF_SCR) if you wish to run multiple runs using a random initialization. These functions will return the run with the smallest error and the results for all runs.

## Method specific information

You must convert X into a non-negative format (either through adding an offset or taking the absolute value) to use ONMTF_SCR.