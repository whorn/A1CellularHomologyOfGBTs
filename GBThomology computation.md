# **Notebook for computing $\mathbb{A}^1$-cellular homology of 2-stage generalized Bott towers**



```python
import numpy as np
```

The package `galois` is used to speed up some computations.


```python
import galois
```

A 2-stage GBT is given by an integer $n$ and a vector $\mathbf{d} = (d_1, \ldots, d_m) \in \mathbb{Z}_2^n$. The following function produces an array of the form  \begin{pmatrix} I_n & 0 & -\mathbf{1}_n & 0  \\ 0 & I_m & \mathbf{d}^T & -\mathbf{1}_m \end{pmatrix}.


```python
def twoStageGBT(n, d_vector):
    #returns the fan for a 2-stage GBT with characterstic vector d.
    m = len(d_vector[0])
    top_half = np.concatenate((np.identity(n),np.zeros([n,m]),np.ones([n,1]),np.zeros([n,1])), axis = 1)
    bottom_half = np.concatenate((np.zeros([m,n]),np.identity(m), d_vector.T, np.ones([m,1])), axis = 1)
    return np.concatenate((top_half, bottom_half))
```

For easier indexing of the shelling of a 2-stage GBT, we will use the fan representation 
\begin{bmatrix} I_n & -\mathbf{1}_n & 0 & 0 \\ 0 & \mathbf{d}^T & I_m & -\mathbf{1}_m \end{bmatrix}
insted. The function `twoStageGBT_alternate` does this.


```python
def twoStageGBT_alternate(n, d_vector):
    #returns an alternatte presentation of the  fan for a 2-stage GBT with characterstic vector d.
    m = len(d_vector[0])
    top_half = np.concatenate((np.identity(n),np.ones([n,1]),np.zeros([n,m+1])), axis = 1)
    bottom_half = np.concatenate((np.zeros([m,n]), d_vector.T ,np.identity(m),np.ones([m,1])), axis = 1)
    return np.concatenate((top_half, bottom_half))
```

Below is an example of applying `twoStageGBT_alternate` to the d-vector $(1,1)$. 


```python
print(twoStageGBT_alternate(2, np.array([[1,1]])))
```

    [[1. 0. 1. 0. 0. 0.]
     [0. 1. 1. 0. 0. 0.]
     [0. 0. 1. 1. 0. 1.]
     [0. 0. 1. 0. 1. 1.]]


The function `shelling_restriction_n_simplex` produces a shelling of the boundary of an $n$-simplex. I returns both an array of faces and a an array of restrictions of each face in the shelling. The function `join_shelling_restriction` does the same for the join $\partial \Delta^n \star \partial \Delta^m$.


```python
def shelling_restriction_n_simplex(n):
    #returns a (n+1)-square matrix with shelling and an (n+1)-square matrix with the restrictions
    return (np.ones(n+1)-np.identity(n+1), np.tril(np.ones(n+1),-1))
shelling,restriction = shelling_restriction_n_simplex(3)

def join_shelling_restriction(n,m):
    shell_n,rest_n = shelling_restriction_n_simplex(n)
    shell_m,rest_m = shelling_restriction_n_simplex(m)
    shell_nm = np.concatenate((np.array([shell_n[0]]),np.array([shell_m[0]])),axis=1)
    rest_nm = np.concatenate((np.array([rest_n[0]]),np.array([rest_m[0]])),axis=1)
    r = len(shell_m)
    for i in range(1,(len(shell_n)*(len(shell_m)))):
        shell_nm = np.concatenate((shell_nm,np.concatenate((np.array([shell_n[i// r]]),np.array([shell_m[i%r]])),axis=1)),axis = 0)
        rest_nm = np.concatenate((rest_nm,np.concatenate((np.array([rest_n[i// r]]),np.array([rest_m[i%r]])),axis=1)),axis = 0)
    return shell_nm, rest_nm

```

Given a fan and a shelling, we can compute the $x$-vectors, and critical simplices. The function `compute_x_vectors`computes the $x$-vectors using a very naive approach, and is slow when $n,m > 5$. The function returns a dictionary where the kets are each $x$-vector, given in binary, with values critical faces in the shelling. 


```python
def compute_x_vectors(fan, shelling,restriction):
    x = {}
    for s in range(len(shelling)):
        face = shelling[s]
        rest = restriction[s]
        for i in range(0, 2**(len(fan.T[0]))):
            b = np.binary_repr(i, width = len(fan.T[0]))
            x_temp = np.array([int(j) for j in list(b)])
            if np.array_equal((x_temp @ fan)*face % 2, rest) :
                if b in x.keys():
                    x[b].append(s)
                else:
                    x[b] = [s]
                break
    return x


```

We use the package `galois` in the function `compute_x_vectors_f2` to considerably speed up this function.


```python
def compute_x_vectors_f2(fan,shelling,restriction):
    x = {}
    for s in range(len(shelling)):
        face = shelling[s]
        rest = restriction[s]
        cols = np.nonzero(face)
        A = galois.GF(2)(fan[:,np.nonzero(face)].astype(int))[:,0,:]
        b = galois.GF(2)(((rest*face)[np.nonzero(face)]).astype(int))
        x_temp = np.linalg.solve(A.T,b)
        b = ""
        for i in x_temp:
            b += str(i)
        if b in x.keys():
            x[b].append(s)
        else:
            x[b] = [s]
    return x


```

Below is an example computing the $x$-vectors of the 2-stage bott tower with $\mathbf{d} = (1)$.


```python
n,m = (1,1)
fan = twoStageGBT_alternate(n, np.array([[1]]))
shell, rest = join_shelling_restriction(n,m)
x = compute_x_vectors_f2(fan, shell,rest)
print("This is the fan:\n ", fan)
print("This is the shellling\n ",shell)
print("This is the restriction \n", rest)
print("x-vectors: ", x)
```

    This is the fan:
      [[1. 1. 0. 0.]
     [0. 1. 1. 1.]]
    This is the shellling
      [[0. 1. 0. 1.]
     [0. 1. 1. 0.]
     [1. 0. 0. 1.]
     [1. 0. 1. 0.]]
    This is the restriction 
     [[0. 0. 0. 0.]
     [0. 0. 1. 0.]
     [1. 0. 0. 0.]
     [1. 0. 1. 0.]]
    x-vectors:  {'00': [0], '11': [1, 3], '10': [2]}


In this example, the dictionary `{'00': [0], '11': [1, 3], '10': [2]}` means: 
- The class based on $r(\sigma_0)$ has no differentials.
- The pair $r(\sigma_1)$ and $r(\sigma_3)$ form a pair of critical simplices and have a differential between each other in the resulting chain complex.
- The class based on $r(\sigma_2)$ has no differentials.

Further computing the weight of each $r(\sigma_i)$, i.e. computing the sum of the corresponding row in the restriction array printed above gives the chain complex
$C_0 = \mathbb{Z}, C_1 = \mathbf{K}^{MW}_1 \oplus \mathbf{K}^{MW}_1 , C_2 = \mathbf{K}^{MW}_2$
and one can compute that the differential $C_2 \to C_1$ is $\begin{pmatrix}\eta & 0 \end{pmatrix}$). Below are some functions that can be used for differental computations. Yielding $H_0 = \mathbb{Z}, H_1 = \mathbf{K}^{MW}_1 \oplus \mathbf{K}^{MW}_1 / \eta, H_2 = { }_\eta \mathbf{K}^{MW}_2$.

## A naive guess of homology groups

In the critical faces of 2-stage GBT's, it seems that only collections of 1, 2, or 4 faces occur. I think these three cases give us the following direct summands in $\mathbb{A}^1$-cellular homology:
- 1 critical face $\sigma_i$ : A summand $\mathbf{K}^{MW}_r(\sigma_i)$ in degree $|r(\sigma_i)|$.
- 2 critcal faces $\sigma_i, \sigma_j$ : In the chain complex they correspond to summands $\mathbf{K}^{MW}_r(\sigma_i)$ \in $C_{|r(\sigma_i)|}$ and  $\mathbf{K}^{MW}_r(\sigma_j)$ \in $C_{|r(\sigma_j)|}$, where $|r(\sigma_j)|= |r(\sigma_i)| +1$ and the differential is $\pm \eta$. This gives summands $ \mathbf{K}^{MW}_{r(\sigma_i)}/\eta \in H^{cell}_{r(\sigma_i)}$ and $_\eta\mathbf{K}^{MW}_{r(\sigma_j)} \in H^{cell}_{r(\sigma_j)}$
- 4 critical faces $\sigma_i, \sigma_j, \sigma_k, \sigma_l$: In this case  $|r(\sigma_j)| -2 = |r(\sigma_j)| = |r(\sigma_k)| = |r(\sigma_i)| +1$. The differentials in this case are $\begin{pmatrix}\pm \eta & \pm \eta \end{pmatrix} : C_{|r(\sigma_i)|+2} \to C_{|r(\sigma_i)|+1}$ and $\begin{pmatrix}\pm \eta \\ \pm \eta \end{pmatrix}: C_{|r(\sigma_i)|+1} \to C_{|r(\sigma_i)|}$. In this case $\pm$ is just whatever sign is needed to make $d^2 = 0$. This gives the homology summands $ \mathbf{K}^{MW}_{r(\sigma_i)}/\eta \in H^{cell}_{r(\sigma_i)}$,  $_\eta\mathbf{K}^{MW}_{|r(\sigma_i)|+1} \oplus \mathbf{K}^{MW}_{|r(\sigma_i)|+1} \in H^{cell}_{|r(\sigma_i)|+1}$ and $_\eta\mathbf{K}^{MW}_{|r(\sigma_i)|+2} \in H^{cell}_{|r(\sigma_i)[+2}$.

Using this naive guess, the function `naive_homology_ranks` returns an array where the first collumn is the amount of $\mathrm{K}_i^{MW}$, the second is the amount of $\mathrm{K}_i^{MW}/\eta$ summands, and the third is the amount of $_\eta \mathrm{K}_i^{MW}$ summands in degree $i$. 


```python
def naive_homology_ranks(x_vectors, restrictions, top_degree):
    x = x_vectors
    res = restrictions
    homology_ranks = np.zeros((top_degree+1,3)).tolist()
    for i in  x.keys():
        if len(x[i])==1:
            homology_ranks[int(np.sum(res[x[i][0]]))][0] += 1
        elif len(x[i])==2:
            homology_ranks[int(np.sum(res[x[i][0]]))][1] += 1
            homology_ranks[int(np.sum(res[x[i][1]]))][2] += 1
        elif len(x[i])==4:
            homology_ranks[int(np.sum(res[x[i][0]]))][1] += 1
            homology_ranks[int(np.sum(res[x[i][0]]))+1][2] += 1
            homology_ranks[int(np.sum(res[x[i][0]]))+1][1] += 1
            homology_ranks[int(np.sum(res[x[i][0]]))+2][2] += 1
    return homology_ranks
```

We try it on our running example and get:


```python
homology = naive_homology_ranks(x, rest, 2)
for i in range(len(homology)):
    print(i,":", homology[i])
```

    0 : [1.0, 0.0, 0.0]
    1 : [1.0, 1.0, 0.0]
    2 : [0.0, 0.0, 1.0]


## Further examples

Below are some more examples. 

We compute the homology of $\mathbb{P}^2 \times \mathbb{P}^2$. 


```python
n,m = (2,2)
fan = twoStageGBT_alternate(n, np.array([[0,0]]))
shell, rest = join_shelling_restriction(n,m)
x = compute_x_vectors_f2(fan, shell,rest)
homology = naive_homology_ranks(x, rest, 4)
for i in range(len(homology)):
    print(i,":", homology[i])
```

    0 : [1.0, 0.0, 0.0]
    1 : [0.0, 2.0, 0.0]
    2 : [0.0, 1.0, 2.0]
    3 : [0.0, 1.0, 1.0]
    4 : [0.0, 0.0, 1.0]


We will now compute the homology for the two other $(2,2)$-GBT's and see that they all have the same homology


```python
n,m = (2,2)
fan = twoStageGBT_alternate(n, np.array([[1,0]]))
shell, rest = join_shelling_restriction(n,m)
x = compute_x_vectors_f2(fan, shell,rest)
homology = naive_homology_ranks(x, rest, 4)
for i in range(len(homology)):
    print(i,":", homology[i])
```

    0 : [1.0, 0.0, 0.0]
    1 : [0.0, 2.0, 0.0]
    2 : [0.0, 1.0, 2.0]
    3 : [0.0, 1.0, 1.0]
    4 : [0.0, 0.0, 1.0]



```python
n,m = (2,2)
fan = twoStageGBT_alternate(n, np.array([[1,1]]))
shell, rest = join_shelling_restriction(n,m)
x = compute_x_vectors_f2(fan, shell,rest)
homology = naive_homology_ranks(x, rest, 4)
for i in range(len(homology)):
    print(i,":", homology[i])
```

    0 : [1.0, 0.0, 0.0]
    1 : [0.0, 2.0, 0.0]
    2 : [0.0, 1.0, 2.0]
    3 : [0.0, 1.0, 1.0]
    4 : [0.0, 0.0, 1.0]


The code can also be used for other shellable fans. Below are the computations for $\mathbb{P}^2$ and $\mathbb{P}^3$.



```python
fanP2 = np.array([[1,0,1],[0,1,1]])
fanP3 = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1]])
shellP2, restP2 = shelling_restriction_n_simplex(2)
shellP3, restP3 = shelling_restriction_n_simplex(3)
xP2 = compute_x_vectors_f2(fanP2, shellP2,restP2)
xP3 = compute_x_vectors_f2(fanP3, shellP3,restP3)
homologyP2 = naive_homology_ranks(xP2, restP2, 2)
homologyP3 = naive_homology_ranks(xP3, restP3, 3)
print("Homology of P2")
for i in range(len(homologyP2)):
    print(i,":", homologyP2[i])

print("Homology of P3")
for i in range(len(homologyP3)):
    print(i,":", homologyP3[i])
```

    Homology of P2
    0 : [1.0, 0.0, 0.0]
    1 : [0.0, 1.0, 0.0]
    2 : [0.0, 0.0, 1.0]
    Homology of P3
    0 : [1.0, 0.0, 0.0]
    1 : [0.0, 1.0, 0.0]
    2 : [0.0, 0.0, 1.0]
    3 : [1.0, 0.0, 0.0]


### Seonjeong's (10,16)-towers

Seonjeong had three different $(10,16)$-towers, below we compute (using the naive homology technique from above) that they all have the same cellular $\mathbb{A}^1$-homology.


```python
n,m = (10,16)
shell,res = join_shelling_restriction(n,m)
fanX = twoStageGBT_alternate(n, np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))
fanY = twoStageGBT_alternate(n, np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))
fanZ = twoStageGBT_alternate(n, np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))
x_vectorX = compute_x_vectors_f2(fanX,shell,res)
x_vectorY = compute_x_vectors_f2(fanY,shell,res)
x_vectorZ = compute_x_vectors_f2(fanZ,shell,res)
homology_ranksX = naive_homology_ranks(x_vectorX,res,n+m)
homology_ranksY = naive_homology_ranks(x_vectorY,res,n+m)
homology_ranksZ = naive_homology_ranks(x_vectorZ,res,n+m)

```


```python
print(homology_ranksX == homology_ranksY)
print(homology_ranksX == homology_ranksZ)
```

    True
    True


## Some not so good code for computing differentials 

In an example above computing differentials was mentioned, here's some code to do that, kinda...


```python
def array_to_str(array):
    s = ""
    for i in array:
        s += str(int(i))
    return s

def differential(simplex_str):
    # returns a dictionary with entries and coefficients.
    d = {}
    for i in range(len(simplex_str)):
        if simplex_str[i] == "1":
            d[simplex_str[0:i] + "0" + simplex_str[i+1:len(simplex_str)]] = (-1)**i
    return d


def differential_reduced(simplex_str, restriction, critical_simplices):
    # returns a dictionary with entries and coefficients, but only the ones in the critical simplices.
    critical_simplex_strings = []
    for i in critical_simplices: 
        critical_simplex_strings.append(array_to_str(res[i]))
    d = {}
    print(critical_simplex_strings)
    for i in range(len(simplex_str)):
        if simplex_str[i] == "1":
            if simplex_str[0:i] + "0" + simplex_str[i+1:len(simplex_str)] in  critical_simplex_strings:
                d[simplex_str[0:i] + "0" + simplex_str[i+1:len(simplex_str)]] = (-1)**i
    return d
```
