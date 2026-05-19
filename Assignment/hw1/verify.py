"""
DSC 206 HW1 — Numerical verification of all problems.
Run: python verify.py
"""
import numpy as np

print("=" * 60)
print("PROBLEM 2 — SVD of A")
print("=" * 60)

A = np.array([[1,2],[-1,2],[1,-2],[-1,-2]], dtype=float)
print("A =\n", A)
print("A^T A =\n", A.T @ A)

# Power method
x = np.array([1., 2.])
B = A.T @ A
for i in range(3):
    x = B @ x
    x = x / np.linalg.norm(x)
    print(f"  iter {i+1}: x = {x}")

# True SVD
U, s, Vt = np.linalg.svd(A)
print(f"True singular values: {s}")
print(f"True V columns (right sing. vecs):\n{Vt.T}")
print(f"True U columns (left sing. vecs):\n{U[:, :2]}")

print("\n" + "=" * 60)
print("PROBLEM 2c — Rotation")
print("=" * 60)
theta = np.pi / 6  # example θ = 30°
Rtheta = np.array([[np.cos(theta), np.sin(theta)],
                   [-np.sin(theta), np.cos(theta)]])
Ahat = A @ Rtheta.T
_, s_hat, Vt_hat = np.linalg.svd(Ahat)
print(f"θ = π/6: σ̂₁ = {s_hat[0]:.4f}  (should equal σ₁ = {s[0]:.4f})")
print(f"First right sing. vec of Â: {Vt_hat[0]}  (should be Rθ v1 = {Rtheta @ Vt.T[:, 0]})")

print("\n" + "=" * 60)
print("PROBLEM 4 — Power method SVD of B")
print("=" * 60)
B4 = np.array([[2.,1.],[1.,3.]])
print("B =\n", B4)
BBT = B4.T @ B4  # = B^2 for symmetric B
x = np.array([1., 1.])
for i in range(3):
    x_new = BBT @ x
    x = x_new / np.linalg.norm(x_new)
    print(f"  iter {i+1}: x = {x}")

U4, s4, Vt4 = np.linalg.svd(B4)
print(f"True singular values: σ₁={(5+np.sqrt(5))/2:.4f}, σ₂={(5-np.sqrt(5))/2:.4f}")
print(f"Computed: {s4}")
print(f"v₁ (estimated): {x}")
print(f"v₁ (true):      {Vt4[0]}")
print(f"σ₁ (Rayleigh estimate): {np.sqrt(x @ BBT @ x):.4f}")

print("\n" + "=" * 60)
print("PROBLEM 5 — M^T M and M M^T")
print("=" * 60)
M = np.array([[1,1],[2,4],[2,4],[3,9]], dtype=float)
print("M^T M =\n", M.T @ M)
print("M M^T =\n", M @ M.T)
evals_MtM = np.linalg.eigvalsh(M.T @ M)
print(f"Eigenvalues of M^T M: {evals_MtM[::-1]}")
print(f"  = 66 ± 4√265: {66+4*np.sqrt(265):.4f}, {66-4*np.sqrt(265):.4f}")
evals_MMt = np.linalg.eigvalsh(M @ M.T)
print(f"Eigenvalues of M M^T: {np.sort(evals_MMt)[::-1]}")

print("\n" + "=" * 60)
print("PROBLEM 6 — Stationary distributions")
print("=" * 60)

# Graph (a)
Pa = np.array([[0.8, 0.2, 0.0, 0.0],
               [0.6, 0.0, 0.4, 0.0],
               [0.0, 0.5, 0.0, 0.5],
               [0.5, 0.0, 0.5, 0.0]])
# Check rows sum to 1
assert np.allclose(Pa.sum(axis=1), 1), "P(a) rows don't sum to 1"
evals_a, evecs_a = np.linalg.eig(Pa.T)
# Find eigenvalue = 1
idx = np.argmin(np.abs(evals_a - 1))
pi_a = np.real(evecs_a[:, idx])
pi_a = pi_a / pi_a.sum()
print(f"Graph (a) stationary dist: {pi_a}")
print(f"  as fractions: [55/82, 15/82, 8/82, 4/82] = {np.array([55,15,8,4])/82}")
print(f"  error = {np.max(np.abs(pi_a - np.array([55,15,8,4])/82)):.2e}")

# Graph (b) — diamond T=0, L=1, R=2, B=3
Pb = np.array([[0.1, 0.3, 0.6, 0.0],
               [0.5, 0.2, 0.0, 0.3],
               [0.5, 0.0, 0.2, 0.3],
               [0.0, 0.6, 0.3, 0.1]])
assert np.allclose(Pb.sum(axis=1), 1), "P(b) rows don't sum to 1"
evals_b, evecs_b = np.linalg.eig(Pb.T)
idx = np.argmin(np.abs(evals_b - 1))
pi_b = np.real(evecs_b[:, idx])
pi_b = pi_b / pi_b.sum()
print(f"\nGraph (b) stationary dist: {pi_b}")
print(f"  as fractions [T,L,R,B] = [40,33,39,24]/136 = {np.array([40,33,39,24])/136}")
print(f"  error = {np.max(np.abs(pi_b - np.array([40,33,39,24])/136)):.2e}")

print("\n" + "=" * 60)
print("PROBLEM 8 — HITS on Figure 2")
print("=" * 60)
# Edges: A->B, A->C, A->D, B->A, B->D, C->A, D->B, D->C
# Nodes: A=0, B=1, C=2, D=3
L = np.array([[0,1,1,1],   # A->B,C,D
              [1,0,0,1],   # B->A,D
              [1,0,0,0],   # C->A
              [0,1,1,0]], dtype=float)  # D->B,C

print("L^T L =\n", (L.T @ L).astype(int))
print("L L^T =\n", (L @ L.T).astype(int))

# Authority = leading eigvec of L^T L
LtL = L.T @ L
evals_auth, evecs_auth = np.linalg.eigh(LtL)
authority = np.abs(evecs_auth[:, -1])
authority /= np.linalg.norm(authority)
print(f"\nAuthority scores: A={authority[0]:.4f}  B={authority[1]:.4f}  C={authority[2]:.4f}  D={authority[3]:.4f}")
print(f"  (leading eigenvalue λ = {evals_auth[-1]:.6f})")

# Hub = leading eigvec of L L^T
LLt = L @ L.T
evals_hub, evecs_hub = np.linalg.eigh(LLt)
hub = np.abs(evecs_hub[:, -1])
hub /= np.linalg.norm(hub)
print(f"Hub scores:       A={hub[0]:.4f}  B={hub[1]:.4f}  C={hub[2]:.4f}  D={hub[3]:.4f}")

# Show power iterations
print("\nPower iterations (unnormalized):")
a = np.ones(4); h = np.ones(4)
for it in range(3):
    a = L.T @ h
    h = L @ a
    print(f"  iter {it+1}: a={a.astype(int).tolist()}  h={h.astype(int).tolist()}")

print("\n" + "=" * 60)
print("PROBLEM 9 — PageRank (corrected graph)")
print("=" * 60)

# Graph: A->B,C,D; B->A,D; C->A; D->B,C  (no dangling nodes)
P9 = np.array([[0,   1/3, 1/3, 1/3],
               [1/2, 0,   0,   1/2],
               [1,   0,   0,   0  ],
               [0,   1/2, 1/2, 0  ]])

# (a) No restart
evals9, evecs9 = np.linalg.eig(P9.T)
idx = np.argmin(np.abs(evals9 - 1))
pi9a = np.real(evecs9[:, idx]); pi9a /= pi9a.sum()
print(f"(a) No restart:       {pi9a}")
print(f"    Expected [1/3, 2/9, 2/9, 2/9] = {np.array([1/3, 2/9, 2/9, 2/9])}")
print(f"    error = {np.max(np.abs(pi9a - np.array([1/3, 2/9, 2/9, 2/9]))):.2e}")

# (b) Restart beta=0.8
beta, n = 0.8, 4
P9b = beta*P9 + (1-beta)/n * np.ones((n, n))
evals9b, evecs9b = np.linalg.eig(P9b.T)
idx = np.argmin(np.abs(evals9b - 1))
pi9b = np.real(evecs9b[:, idx]); pi9b /= pi9b.sum()
print(f"\n(b) Restart 0.2:      {pi9b}")
print(f"    Expected [9/28, 19/84, 19/84, 19/84] = {np.array([9/28, 19/84, 19/84, 19/84])}")
print(f"    error = {np.max(np.abs(pi9b - np.array([9/28, 19/84, 19/84, 19/84]))):.2e}")

# (c) Add 4 spam nodes v1..v4 -> A, n=8, beta=0.8
n8 = 8
P9c = np.zeros((n8, n8))
P9c[0,1]=1/3; P9c[0,2]=1/3; P9c[0,3]=1/3  # A->B,C,D
P9c[1,0]=1/2; P9c[1,3]=1/2                  # B->A,D
P9c[2,0]=1.0                                 # C->A
P9c[3,1]=1/2; P9c[3,2]=1/2                  # D->B,C
for i in range(4, 8): P9c[i, 0] = 1.0       # vi->A

P9c_r = beta*P9c + (1-beta)/n8 * np.ones((n8, n8))
evals9c, evecs9c = np.linalg.eig(P9c_r.T)
idx = np.argmin(np.abs(evals9c - 1))
pi9c = np.real(evecs9c[:, idx]); pi9c /= pi9c.sum()
print(f"\n(c) 4 spam nodes, restart 0.2:")
print(f"    π_A={pi9c[0]:.6f}  π_B={pi9c[1]:.6f}  π_C={pi9c[2]:.6f}  π_D={pi9c[3]:.6f}  π_vi={pi9c[4]:.6f}")
print(f"    Expected [93,53,53,53,7]/280 = {np.array([93,53,53,53,7])/280}")
print(f"    error = {np.max(np.abs(pi9c[:4] - np.array([93,53,53,53])/280)):.2e}")

print("\nAll verifications done.")
