# Answers 4

## Time complexity of learning

- The time complexity comes out to be **O((M+1)NCI)**, where N = number of samples, M = number of features,
C = number of classes (for binary-class C=1), I = number of iterations or epochs

## Time complexity of prediction

- It is **O(NC(M+1))** with the same terminology mentioned above.

## Space complexity of learning

- there are 4 matrices or vectors need to be stored which are X (NxM), y (Nx1), Weights (MxC) with addition of bias. So the space complexity becomes **O(NM + N + MC)**

## Space complexity of prediction

- For prediction we need only weights in the memory so the space complexity is order of the weights that is **O(MC)**