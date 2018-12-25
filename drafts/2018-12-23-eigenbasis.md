There are four prerequisites to understand Eigenbasis, i.e. linear transformations, determinants, linear systems and change of basis. 

### the meaning of eigenvector and eigenvalue

#### linear transformation
Multiplying by a matrix can be interpreted as a linear transformation, which transforms a vector from the original coordinate into a new coordinate whose basis vectors are the column vectors of the matrix. 
The column space of a matrix can be seen as a new basis expressed in the original coordinate. For example, $$\begin{bmatrix}3 & 1 \ 0 & 2 \end{bmatrix}$$ represents a new basis with [3, 0] and [1, 2] being the new basis vector. And both of these two vectors are expressed in original coordinate. Then a vector [1, 1] in the new coordinate will be the same vector as [4, 2] in the original coordinate. 

Eigenvector is the vector in the space, which remains on its own span after linear transformation. For example, all vectors on x-axis will still lie on x-axis after applying the linear transformation of $$\begin{bmatrix}3 & 1 \ 0 & 2 \end{bmatrix}$$. That means, all vectors on x-axis are eigenvectors of the matrix.

Then let's look at the definition function of eigenvector and eigenvalue:
$$
Av = \lambda v
$$
From above, we can see that the left part of the equation results a vector with the same direction. That explains why a matrix-vector product equals to a scalar-vector multiplication. Then it's easy to understand, that eigenvalue is just a scalar that scales the length of the vector.

### calculation of eigenvector and eigenvalue

$$
\lambda v = \begin{bmatrix}\lambda & 0 & 0 \\ 0 & \lambda & 0 \\ 0 & 0 & \lambda \end{bmatrix}v = \lambda Iv
$$

$$
(A-\lambda I) v = 0
$$

To make sure the equation always valid for any &&v&&, &&det(A - \lambda I) = 0&&.

There are some points that I want to emphasize:

* eigenvalues which are complex numbers generally correspond to some kind of rotation in the transformation. For example, &&\begin{bmatrix}0 & -1 \ 1 & 0\end{bmatrix}&&

* a single eigenvalue can have more that a line full of eigenvectors. For example, scaling everything by 2
  $$
  A = \begin{bmatrix}2 & 0 \\ 0 & 2 \end{bmatrix}
  $$




### Eigenbasis

Eigenbasis means all basis vectors are eigenvectors. Before looking at eigenbasis, let's have a look at diagonal matrix first.

Diagonal matrix is a kind of matrix, which all non-zero entries are in the diagonal line. For example, [3 0 0; 0 2 0; 0 0 5] is a diagonal matrix. Diagonal matrix has some adorable features that may make our life better, one of which is that the product of two diagonal matrices is still a diagonal matrix. For example, [3 0 0; 0 2 0; 0 0 5] * [2 0 0; 0 1 0; 0 0 1] = [6 0 0; 0 2 0; 0 0 5]. Then if we multiply the diagonal matrix by itself n times, the product will just be the [a^n 0 0; 0 b^n 0; 0 0 c^n].

So now the problem is how do we get eigenbasis. If we want to perform a transformation, which is represented by a matrix in the original coordinate. We want to express the same transformation but in the eigenbasis instead. 

#### change of basis
When we have a vector expressed in original coordinate and a matrix whose column space represents a new coordinate, how can we get the new expression of this vector in the new coordinate? 

As we've discussed in linear transformation, Av_A translates a the coordinate of the vector from a new coordinate into the original one. If Av_A = v, then we get v_A = A^-1v. v_A is the coordinate which is expressed in the new coordinate.

#### two ways to describe a linear transformation
* column vectors as basis vectors
* eigenvectors and eigenvalues. So Av can also describe a linear transformation

We want the linear transformation to happen in the eigen space where eigen vectors are basis vectors instead of original space. But instead of translate the linear transformation itself, we translate Av, because Av = lambda v which implies the linear transformation.  Then we get (v^-1Av)^n = v^-1A^nv = X. So A^n = vXv^-1. That's how we translate the n times linear transformation back to original coordinate.