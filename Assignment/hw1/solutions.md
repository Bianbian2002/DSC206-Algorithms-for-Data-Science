# DSC 206 — Homework 1 Solutions
Zeyu Bian, A69041100
---

## Problem 1 (5 pts)

**Claim:** For the rank-$k$ approximation

$$
A_k = \sum_{i=1}^k \sigma_i u_i v_i^T,
$$

we have

$$
\|(A - A_k)x\|_2 \le \sigma_{k+1}\|x\|_2.
$$

**Proof.**

$A - A_k = \sum_{i=k+1}^r \sigma_i u_i v_i^T$, where $r = \operatorname{rank}(A)$.

Expand $x$ in the orthonormal basis $\{v_1, \ldots, v_n\}$:

$$
x = \sum_j (v_j^T x)\, v_j.
$$

Then
$$
(A - A_k)x = \sum_{i=k+1}^r \sigma_i u_i v_i^T \Bigl(\sum_j (v_j^T x) v_j\Bigr)
= \sum_{i=k+1}^r \sigma_i (v_i^T x) u_i.
$$

Since $\{u_i\}$ are orthonormal,
$$
\|(A-A_k)x\|_2^2 = \sum_{i=k+1}^r \sigma_i^2 (v_i^T x)^2
\le \sigma_{k+1}^2 \sum_{i=k+1}^r (v_i^T x)^2
\le \sigma_{k+1}^2 \sum_j (v_j^T x)^2
= \sigma_{k+1}^2 \|x\|_2^2.
$$

The first inequality uses $\sigma_i \le \sigma_{k+1}$ for $i \ge k+1$; the second uses Parseval's identity. Taking square roots gives the result. $\square$

**Key insight:** The residual $A - A_k$ only acts on the tail singular directions $\{v_{k+1}, \ldots\}$, and its operator norm equals $\sigma_{k+1}$, the largest discarded singular value.

---

## Problem 2 (15 pts)

$$
A = \begin{pmatrix}
1&2\\
-1&2\\
1&-2\\
-1&-2
\end{pmatrix}
$$

### (a) Power method, $k=3$ steps from $x^{(0)}=[1,2]^T$

First compute $B = A^T A$:

$$
A^T A
= \begin{pmatrix}1&-1&1&-1\\2&2&-2&-2\end{pmatrix}
\begin{pmatrix}1&2\\-1&2\\1&-2\\-1&-2\end{pmatrix}
= \begin{pmatrix}4&0\\0&16\end{pmatrix}.
$$

Power iteration: $x^{(t+1)} \propto Bx^{(t)}$.

Start from

$$
x^{(0)} = \begin{pmatrix}1\\2\end{pmatrix}.
$$

Step 1:

$$
Bx^{(0)} = \begin{pmatrix}4\\32\end{pmatrix},
\qquad
x^{(1)} = \frac{1}{\sqrt{65}}\begin{pmatrix}1\\8\end{pmatrix}.
$$

Step 2:

$$
Bx^{(1)} = \frac{1}{\sqrt{65}}\begin{pmatrix}4\\128\end{pmatrix},
\qquad
x^{(2)} = \frac{1}{\sqrt{1025}}\begin{pmatrix}1\\32\end{pmatrix}.
$$

Step 3:

$$
Bx^{(2)} = \frac{1}{\sqrt{1025}}\begin{pmatrix}4\\512\end{pmatrix},
\qquad
x^{(3)} = \frac{1}{\sqrt{16385}}\begin{pmatrix}1\\128\end{pmatrix}.
$$

After 3 steps: $x^{(3)} = [1,128]^T/\sqrt{16385}$.

**Convergence note:** The ratio of components $128/1 = 128 = 2^7$ is growing toward $\infty$, converging to $v_1 = [0,1]^T$ (see part b).

### (b) Actual SVD of $A$

Since $A^TA = \operatorname{diag}(4,16)$, the eigensystem is immediate.

For $i=1$,

$$
\sigma_1 = \sqrt{16} = 4,
\qquad
v_1 = \begin{pmatrix}0\\1\end{pmatrix},
\qquad
u_1 = \frac{Av_1}{\sigma_1}
= \frac{1}{4}\begin{pmatrix}2\\2\\-2\\-2\end{pmatrix}
= \begin{pmatrix}\tfrac12\\\tfrac12\\-\tfrac12\\-\tfrac12\end{pmatrix}.
$$

For $i=2$,

$$
\sigma_2 = \sqrt{4} = 2,
\qquad
v_2 = \begin{pmatrix}1\\0\end{pmatrix},
\qquad
u_2 = \frac{Av_2}{\sigma_2}
= \frac{1}{2}\begin{pmatrix}1\\-1\\1\\-1\end{pmatrix}
= \begin{pmatrix}\tfrac12\\-\tfrac12\\\tfrac12\\-\tfrac12\end{pmatrix}.
$$

### (c) Rotation

Let

$$
R_\theta =
\begin{pmatrix}
\cos\theta&\sin\theta\\
-\sin\theta&\cos\theta
\end{pmatrix}
$$

(clockwise rotation).

**c.1** The problem defines $\hat A^T = R_\theta A^T$, so $\hat A = A R_\theta^T$.

$$
\hat A
= A\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}
= \begin{pmatrix}
\cos\theta+2\sin\theta & -\sin\theta+2\cos\theta\\
-\cos\theta+2\sin\theta & \sin\theta+2\cos\theta\\
\cos\theta-2\sin\theta & -\sin\theta-2\cos\theta\\
-\cos\theta-2\sin\theta & \sin\theta-2\cos\theta
\end{pmatrix}.
$$

**c.2** Write $A = U\Sigma V^T$; then $\hat A = U\Sigma(R_\theta V)^T$.

Thus **the singular values are unchanged**: $\hat\sigma_1 = 4$, $\hat\sigma_2 = 2$.

The **new first right singular vector** is

$$
\hat v_1 = R_\theta v_1 = R_\theta\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}\sin\theta\\\cos\theta\end{pmatrix}.
$$

In general, the singular values are rotation-invariant, and the right singular vectors rotate with the data.

---

## Problem 3 (10 pts)

Let $A$ be $m\times n$ with unit-norm rows $\{a_i\}$.

### Part 1 — single synthetic document

We want $y$ (unit norm) maximizing

$$
\sum_i (a_i^T y)^2 = \|Ay\|_2^2 = y^T A^T A\, y.
$$

By the Rayleigh quotient, the maximum is the **largest eigenvalue** of $A^TA$, achieved at its leading eigenvector, which is the **first right singular vector $v_1$** of $A$.

*This $v_1$ is the principal direction of the document cloud; it represents the topic most coherently expressed across all documents.*

### Part 2 — $k$ orthogonal synthetic documents

Maximize

$$
\sum_{j=1}^k \|Ay_j\|_2^2 = \sum_{j=1}^k y_j^T A^T A\, y_j
$$

subject to $\{y_j\}$ orthonormal.

By the variational characterization of eigenvalues (Fan trace maximization),

$$
\max_{\{y_j\}\text{ o.n.}} \sum_{j=1}^k y_j^T(A^TA)y_j
= \lambda_1+\lambda_2+\cdots+\lambda_k,
$$

achieved at the **first $k$ right singular vectors** $\{v_1, \ldots, v_k\}$.

These are also the first $k$ columns of $V$ in the SVD $A = U\Sigma V^T$.

---

## Problem 4 (5 pts)

$$
B = \begin{pmatrix}2&1\\1&3\end{pmatrix}, \quad x^{(0)} = [1,1]^T.
$$

$B^T B = B^2$ (since $B$ is symmetric):

$$
B^2 = \begin{pmatrix}5&5\\5&10\end{pmatrix}.
$$

### Three power iterations

Starting from

$$
x^{(0)} = \begin{pmatrix}1\\1\end{pmatrix},
$$

the first three iterations are:

$$
B^2x^{(0)} = \begin{pmatrix}10\\15\end{pmatrix},
\qquad
x^{(1)} = \frac{1}{\sqrt{13}}\begin{pmatrix}2\\3\end{pmatrix},
$$

$$
B^2x^{(1)} = \frac{1}{\sqrt{13}}\begin{pmatrix}25\\40\end{pmatrix}
= \frac{5}{\sqrt{13}}\begin{pmatrix}5\\8\end{pmatrix},
\qquad
x^{(2)} = \frac{1}{\sqrt{89}}\begin{pmatrix}5\\8\end{pmatrix},
$$

$$
B^2x^{(2)} = \frac{5}{\sqrt{89}}\begin{pmatrix}13\\21\end{pmatrix},
\qquad
x^{(3)} = \frac{1}{\sqrt{610}}\begin{pmatrix}13\\21\end{pmatrix}.
$$

**First right singular vector estimate:** $v_1 \approx [13,21]^T/\sqrt{610}$.

(Ratio $21/13 \approx 1.615 \approx \varphi = (1+\sqrt5)/2$, converging rapidly.)

### First singular value

$$
\sigma_1 \approx \|Bx^{(3)}\|_2
= \left\|B\frac{[13,21]^T}{\sqrt{610}}\right\|_2
= \frac{\|[47,76]^T\|_2}{\sqrt{610}}
= \sqrt{\frac{7985}{610}}
\approx 3.618.
$$

Exact: eigenvalues of $B$ are $\lambda = \frac{5\pm\sqrt5}{2}$, so $\sigma_1 = \frac{5+\sqrt5}{2}$.

### First left singular vector

$$
u_1 = \frac{Bv_1}{\sigma_1} \approx \frac{[47,76]^T}{\sqrt{7985}}.$$

Exact: $u_1 = v_1$ (since $B$ is symmetric positive definite, $U=V$ in its SVD).

### Second singular vector

$v_2 \perp v_1$ and is the other eigenvector:

$$
v_2 = \frac{[-(1+\sqrt5),\,2]^T}{\sqrt{10+2\sqrt5}},
\qquad
\sigma_2 = \frac{5-\sqrt5}{2} \approx 1.382.
$$

### Final SVD

$$
B = U\Sigma V^T,
\quad
U = V = \frac{1}{\sqrt{10+2\sqrt5}}
\begin{pmatrix}
2 & -(1+\sqrt5)\\
1+\sqrt5 & 2
\end{pmatrix},
\quad
\Sigma = \begin{pmatrix}\frac{5+\sqrt5}{2}&0\\0&\frac{5-\sqrt5}{2}\end{pmatrix}.
$$

---

## Problem 5 (12 pts)

$$
M = \begin{pmatrix}
1&1\\
2&4\\
2&4\\
3&9
\end{pmatrix}.
$$

### (a) $M^TM$ and $MM^T$

$$
M^TM
= \begin{pmatrix}
1^2+2^2+2^2+3^2 & 1\cdot1+2\cdot4+2\cdot4+3\cdot9\\
\cdot & 1^2+4^2+4^2+9^2
\end{pmatrix}
= \begin{pmatrix}18&44\\44&114\end{pmatrix}.
$$

$MM^T$ is $4\times 4$ with $(i,j)$ entry equal to $m_i \cdot m_j$:

$$
MM^T = \begin{pmatrix}
2&6&6&12\\
6&20&20&42\\
6&20&20&42\\
12&42&42&90
\end{pmatrix}.
$$

*(Rows 2 and 3 of $M$ are identical, so rows and columns 2 and 3 of $MM^T$ are identical.)*

### (b) Eigenpairs of $M^TM$

$$
\det(M^TM - \lambda I) = \lambda^2 - 132\lambda + 116 = 0.
$$

$$
\lambda = 66 \pm 4\sqrt{265}.
$$

For $\lambda_1 = 66+4\sqrt{265}$, solve $(18-\lambda_1)v_1 + 44v_2 = 0$, giving

$$
v_1 \propto \begin{pmatrix}11\\12+\sqrt{265}\end{pmatrix},
\quad
\text{normalized by } \sqrt{530+24\sqrt{265}}.
$$

For $\lambda_2 = 66-4\sqrt{265}$,

$$
v_2 \propto \begin{pmatrix}11\\12-\sqrt{265}\end{pmatrix},
\quad
\text{normalized by } \sqrt{530-24\sqrt{265}}.
$$

*(Note $\sqrt{265}\approx 16.28$, so $\lambda_1\approx 131.1$, $\lambda_2\approx 0.88$, and $v_2$ has a negative second component.)*

### (c) Eigenvalues of $MM^T$

$MM^T$ (a $4\times4$ matrix of rank at most $2$) shares its **nonzero** eigenvalues with $M^TM$. Therefore,

$$
\text{eigenvalues of } MM^T = \{66+4\sqrt{265},\; 66-4\sqrt{265},\; 0,\; 0\}.
$$

---

## Problem 6 (6 pts)

### Network (a)

**Graph:** 4-node chain with wrap-around (reading the edge labels in Figure 1):

- Node 1: $P(1\to1)=0.8$, $P(1\to2)=0.2$
- Node 2: $P(2\to1)=0.6$, $P(2\to3)=0.4$
- Node 3: $P(3\to2)=0.5$, $P(3\to4)=0.5$
- Node 4: $P(4\to3)=0.5$, $P(4\to1)=0.5$

**Solve $\pi P = \pi$:**

From $\pi_4 = 0.5\pi_3$ and $0.75\pi_3 = 0.4\pi_2$ and $(11/15)\pi_2 = (1/5)\pi_1$:

$$\pi_2 = \frac{3}{11}\pi_1, \quad \pi_3 = \frac{8}{55}\pi_1, \quad \pi_4 = \frac{4}{55}\pi_1.$$

Normalizing ($\pi_1\cdot\frac{82}{55}=1$):

$$\boxed{\pi_1 = \frac{55}{82},\quad \pi_2 = \frac{15}{82},\quad \pi_3 = \frac{8}{82},\quad \pi_4 = \frac{4}{82}.}$$

### Network (b)

**Graph:** 4-node diamond (T=top, L=left, R=right, B=bottom):

- T: $P(T\to T)=0.1$, $P(T\to L)=0.3$, $P(T\to R)=0.6$
- L: $P(L\to T)=0.5$, $P(L\to L)=0.2$, $P(L\to B)=0.3$
- R: $P(R\to T)=0.5$, $P(R\to R)=0.2$, $P(R\to B)=0.3$
- B: $P(B\to L)=0.6$, $P(B\to R)=0.3$, $P(B\to B)=0.1$

**Solve $\pi P = \pi$:**

From equations:
$$0.9\pi_T = 0.5(\pi_L+\pi_R), \quad 0.9\pi_B = 0.3(\pi_L+\pi_R)$$
so $\pi_B = \frac{1}{3}\cdot\frac{0.9}{0.5}\cdot\pi_T \cdot\frac{1}{2}$... more directly: $\pi_B = 0.6\pi_T$.

Then $0.8\pi_L = 0.66\pi_T$ giving $\pi_L = \frac{33}{40}\pi_T$ and $0.8\pi_R = 0.78\pi_T$ giving $\pi_R = \frac{39}{40}\pi_T$.

Normalizing ($\pi_T\cdot\frac{17}{5}=1$... wait: $1+\frac{33}{40}+\frac{39}{40}+\frac{3}{5} = \frac{40+33+39+24}{40}=\frac{136}{40}=\frac{17}{5}$):

$$\boxed{\pi_T = \frac{5}{17} = \frac{40}{136},\quad \pi_L = \frac{33}{136},\quad \pi_R = \frac{39}{136},\quad \pi_B = \frac{3}{17} = \frac{24}{136}.}$$

---

## Problem 7 (4 pts)

**Example:** 2-state Markov chain on $\{0,1\}$ with $P(0\to1)=1$ and $P(1\to0)=1$.

Transition matrix:

$$
Q = \begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

This is strongly connected (you can reach any state from any state).

**Initial distribution:** $q^{(0)} = [1,0]^T$ (start at state 0 with certainty).

Then $q^{(t)} = Q^t q^{(0)}$:

$$
q^{(0)}=[1,0]^T,
\quad
q^{(1)}=[0,1]^T,
\quad
q^{(2)}=[1,0]^T,
\quad
\ldots
$$

$q^{(t)}$ alternates between $[1,0]^T$ and $[0,1]^T$ forever, so $\lim_{t\to\infty} q^{(t)}$ does **not** exist.

**Why?** The chain is **periodic** with period 2. At even times all mass is at state 0; at odd times all mass is at state 1. The Fundamental Theorem requires the chain to be *aperiodic* (as well as irreducible) for $q^{(t)}$ to converge. While the long-run average $\bar q = \frac12[1,0]^T + \frac12[0,1]^T = [1/2,1/2]^T$ converges to the unique stationary distribution $\pi=[1/2,1/2]^T$, the sequence itself oscillates.

---

## Problem 8 (5 pts) — HITS on Figure 2

**Graph** (from Figure 2): $A\to B$, $A\to C$, $A\to D$, $B\to A$, $B\to D$, $C\to A$, $D\to B$, $D\to C$.

Out-degrees: $A=3$, $B=2$, $C=1$, $D=2$. In-degrees: all equal 2.

Adjacency matrix $L$ (rows = from, cols = to, order $A,B,C,D$):

$$
L = \begin{pmatrix}
0&1&1&1\\
1&0&0&1\\
1&0&0&0\\
0&1&1&0
\end{pmatrix}.
$$

### Authority scores — leading eigenvector of $L^TL$

$$
L^TL = \begin{pmatrix}
2&0&0&1\\
0&2&2&1\\
0&2&2&1\\
1&1&1&2
\end{pmatrix}.
$$

*Note:* rows and columns $B$ and $C$ are identical, so $a_B = a_C$ in the leading eigenvector.

**Antisymmetric eigenvector** ($a_A=a_D=0$, $a_B=-a_C$): gives eigenvalue $\lambda=0$.

**Symmetric subspace** ($a_B=a_C=p$, unknowns $a_A,p,a_D$): the characteristic equation reduces to

$$
\lambda^3 - 8\lambda^2 + 17\lambda - 8 = 0,
$$

with roots $\lambda \approx 4.814,\ 2.529,\ 0.657$ (no rational roots).

For the leading eigenvalue $\lambda_1 \approx 4.814$, the eigenvector satisfies $a_D=(\lambda_1-2)a_A$ and $p=(\lambda_1-2)a_A/(\lambda_1-4)$. Normalizing,

$$
a \approx \begin{pmatrix}0.1745\\0.6035\\0.6035\\0.4910\end{pmatrix}
\quad
(A,B,C,D).
$$

### Hub scores — leading eigenvector of $LL^T$

$$
LL^T = \begin{pmatrix}
3&1&0&2\\
1&2&1&0\\
0&1&1&0\\
2&0&0&2
\end{pmatrix}.
$$

Same nonzero eigenvalues as $L^TL$ (both share $\lambda_1\approx 4.814$). The leading eigenvector is

$$
h \approx \begin{pmatrix}0.7739\\0.3033\\0.0795\\0.5501\end{pmatrix}
\quad
(A,B,C,D).
$$

### Power iteration (first 3 rounds, unnormalized)

Update rule: $a \leftarrow L^T h$ then $h \leftarrow L a$ (normalize after convergence).

Start with

$$
a^{(0)} = h^{(0)} = \begin{pmatrix}1\\1\\1\\1\end{pmatrix}.
$$

Then the first three unnormalized rounds are

$$
a^{(1)} = \begin{pmatrix}2\\2\\2\\2\end{pmatrix},
\qquad
h^{(1)} = \begin{pmatrix}6\\4\\2\\4\end{pmatrix},
$$

$$
a^{(2)} = \begin{pmatrix}6\\10\\10\\10\end{pmatrix},
\qquad
h^{(2)} = \begin{pmatrix}30\\16\\6\\20\end{pmatrix},
$$

$$
a^{(3)} = \begin{pmatrix}22\\50\\50\\46\end{pmatrix},
\qquad
h^{(3)} = \begin{pmatrix}146\\68\\22\\100\end{pmatrix}.
$$

After iteration 3, the ratios $a_B:a_C:a_D:a_A = 50:50:46:22$ are converging toward the eigenvector $B=C > D \gg A$, and $h_A:h_D:h_B:h_C = 146:100:68:22$ shows $A \gg D > B \gg C$.

**Interpretation:**

- **$A$** has by far the highest hubbiness (out-degree 3, points to all others).
- **$B$ and $C$** tie for highest authority: each is pointed to by 2 nodes ($A$ and $D$), and $D$ itself has high hubbiness.
- **$D$** has the second-highest hubbiness (out-degree 2, points to the high-authority nodes $B,C$) and second-highest authority (pointed to by $A$ and $B$).
- **$C$** has the lowest hubbiness (out-degree 1, only points to $A$).

---

## Problem 9 (15 pts) — PageRank on Figure 2

**Same graph**: $A\to B$, $A\to C$, $A\to D$, $B\to A$, $B\to D$, $C\to A$, $D\to B$, $D\to C$.

Every node has at least one outgoing edge, so there are **no dangling nodes**.

**Transition matrix** (row-stochastic):

$$
P = \begin{pmatrix}
0&\tfrac13&\tfrac13&\tfrac13\\
\tfrac12&0&0&\tfrac12\\
1&0&0&0\\
0&\tfrac12&\tfrac12&0
\end{pmatrix}.
$$

### (a) PageRank without restart

Solve $\pi P = \pi$ (i.e. $\pi_j = \sum_i \pi_i P_{ij}$):

$$
\pi_A = \tfrac12\pi_B + \pi_C,
\quad
\pi_B = \tfrac13\pi_A + \tfrac12\pi_D,
\quad
\pi_C = \tfrac13\pi_A + \tfrac12\pi_D,
\quad
\pi_D = \tfrac13\pi_A + \tfrac12\pi_B.
$$

From the equations for $\pi_B$ and $\pi_C$, we get $\pi_B = \pi_C$. Then the symmetry between the equations for $\pi_B$ and $\pi_D$ gives $\pi_B = \pi_D$ as well.

Setting $b = \pi_B = \pi_C = \pi_D$, we get $\pi_A = \tfrac12 b + b = \tfrac32 b$. Normalizing,

$$
\tfrac32 b + b + b + b = \tfrac92 b = 1
\implies
b = \tfrac29.
$$

$$
\boxed{\pi_A = \frac{1}{3},\quad \pi_B = \pi_C = \pi_D = \frac{2}{9}.}
$$

$A$ has the highest PageRank, reflecting its role as the central hub with the highest out-degree. $B$, $C$, and $D$ are equal by symmetry.

### (b) PageRank with restart, $1-\beta = 0.20$ ($\beta = \tfrac45$, $n=4$)

$$
\pi_j = \tfrac45\sum_i \pi_i P_{ij} + \tfrac{1}{20}.
$$

By the same symmetry argument as in part (a), let $b = \pi_B = \pi_C = \pi_D$. Then

$$
\pi_A = \tfrac45\left(\tfrac{b}{2} + b\right) + \tfrac{1}{20} = \tfrac65 b + \tfrac{1}{20},
$$

and

$$
b = \tfrac45\left(\tfrac{\pi_A}{3} + \tfrac{b}{2}\right) + \tfrac{1}{20} = \tfrac{4\pi_A}{15} + \tfrac{2b}{5} + \tfrac{1}{20}.
$$

Substituting $\pi_A = \tfrac65 b + \tfrac{1}{20}$ gives

$$
b = \tfrac{4}{15}\left(\tfrac65 b + \tfrac{1}{20}\right) + \tfrac{2b}{5} + \tfrac{1}{20}
= \tfrac{8b}{25} + \tfrac{1}{75} + \tfrac{2b}{5} + \tfrac{1}{20}.
$$

Hence

$$
b\left(1 - \tfrac{8}{25} - \tfrac{2}{5}\right) = \tfrac{1}{75} + \tfrac{1}{20}
= \tfrac{19}{300},
$$

so

$$
\tfrac{7b}{25} = \tfrac{19}{300}
\implies
b = \tfrac{19}{84}.
$$

Therefore,

$$
\boxed{\pi_A = \frac{9}{28},\quad \pi_B = \pi_C = \pi_D = \frac{19}{84}.}
$$

### (c) Add 4 spam nodes $v_1,v_2,v_3,v_4$ with $v_i\to A$; restart $1-\beta=0.20$, $n=8$

Now $(1-\beta)/n = \tfrac{1}{40}$. Since no one links to any $v_i$,

$$
\pi_{v_i} = \tfrac45\cdot 0 + \tfrac{1}{40} = \tfrac{1}{40}
\quad
\text{for each } i.
$$

Again let $b = \pi_B = \pi_C = \pi_D$. Then

$$
\pi_A = \tfrac45\left(\tfrac{b}{2}+b+4\cdot\tfrac{1}{40}\right)+\tfrac{1}{40}
= \tfrac65 b + \tfrac{21}{200},
$$

and

$$
b = \tfrac{4\pi_A}{15}+\tfrac{2b}{5}+\tfrac{1}{40}.
$$

Substituting for $\pi_A$ gives

$$
\tfrac{7b}{25} = \tfrac{4}{15}\left(\tfrac65 b+\tfrac{21}{200}\right)-\tfrac{2b}{5}+\tfrac{1}{40}
= \tfrac{7}{250}+\tfrac{1}{40}
= \tfrac{53}{1000}.
$$

Thus

$$
b = \tfrac{53}{1000}\cdot\tfrac{25}{7} = \tfrac{53}{280},
$$

and

$$
\pi_A = \tfrac65\cdot\tfrac{53}{280}+\tfrac{21}{200}
= \tfrac{318}{1400}+\tfrac{147}{1400}
= \tfrac{93}{280}.
$$

So

$$
\boxed{\pi_A = \frac{93}{280},\quad \pi_B = \pi_C = \pi_D = \frac{53}{280},\quad \pi_{v_i} = \frac{1}{40} = \frac{7}{280}.}
$$

**Effect of link farm:** $A$'s PageRank increased from $\tfrac{9}{28}\approx 0.321$ to $\tfrac{93}{280}\approx 0.332$, only a modest gain. $B,C,D$ dropped from $\tfrac{19}{84}\approx 0.226$ to $\tfrac{53}{280}\approx 0.189$ because the spam nodes absorb teleportation mass without contributing useful links to anyone but $A$.