#!/usr/bin/env python3
# sqL1_box_admm_compare.py
# ---------------------------------------------------------------
import numpy as np, time, csv, datetime, argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt

# ===============================================================
# 0.  squared-ℓ¹ objective
# ===============================================================
def F(x, v, xk, rho):
    """F(x)=½‖x-v‖₂² + ½ρ‖x - xk‖₁²"""
    return 0.5*np.linalg.norm(x - v, 2)**2 + 0.5*rho*np.linalg.norm(x - xk, 1)**2 # Yue: I changed this and added xk


def prox_sqL1_unconstrained(q, rho_t, tol=1e-12): # Yue: want to test if admm will be faster by using this.
    if rho_t == 0:
        return q.copy()
    lo, hi = 0.0, np.max(np.abs(q))
    while hi - lo > tol:
        lam = 0.5*(lo + hi)
        s   = np.sum(np.maximum(np.abs(q) - lam, 0.0))
        phi = lam - rho_t * s    # root test
        if phi > 0:
            hi = lam
        else:
            lo = lam
    lam = 0.5*(lo + hi) 
    return np.sign(q) * np.maximum(np.abs(q) - lam, 0.0)

def prox_1(v, rho):
	d, ll = np.size(v), np.linalg.norm(v, ord = 1)
	x, summ = np.zeros(np.shape(v)), np.zeros(d+1)
	idx = np.argsort(np.abs(v), axis = 0)
	for i in range(1, d+1):
		summ[d - i] = summ[d - i + 1] + np.abs(v[idx[d - i], 0])
	# print(summ, idx)
	# print(v[idx[:], 0])
	if ll == 0:  # l1_norm of z = 0(or say >= v(d))
		return x
	if ll * rho / (d*rho + 1) <= np.abs(v[idx[0], 0]): # l1_norm of z < v(0)
		for i in range(d):
			x[idx[i], 0] = v[idx[i], 0] - np.sign(v[idx[i], 0]) * ll * rho / (d*rho + 1)
	for i in range(1, d):
		dd = summ[i] * rho / ((d - i)*rho + 1)
		# print(dd)
		if dd > np.abs(v[idx[i-1], 0]) and dd <= np.abs(v[idx[i], 0]):
		# v(i-1) <= l1_norm of z < v(i), note that it can not have "=" on both sides
			for t in range(i, d):
				x[idx[t], 0] = v[idx[t], 0] - np.sign(v[idx[t], 0]) * summ[i] * rho / ((d - i)*rho + 1)
			# break
	return x

def prox_box(v, psi):
	return np.where(np.abs(v) <= psi, v, psi * np.sign(v))

def prox_1_admm_box(v, xk, rho, R, beta, epsi_hat, max_ti, thres):
	start_time = time.time()
	x, y, lamb = np.zeros(np.shape(v)), np.zeros(np.shape(v)), 0
	oldx, oldy, oldlamb = x, y, lamb
	while True:
		now = time.time()
		if now - start_time > max_ti:
			break
		x = prox_box((v + beta * (y + xk) - lamb) / (beta + 1), R)
		#y = prox_1((beta * (x - xk) + lamb) / beta, rho / beta)
		y = prox_sqL1_unconstrained((beta * (x - xk) + lamb) / beta, rho / beta) # Yue: Use new algorithm
		lamb = lamb + beta * (x - y - xk)
		# if np.linalg.norm(x - y - xk) <= epsi_hat and beta * np.linalg.norm(y - oldy) <= epsi_hat:
		# 	return x
		if F(x, v, xk, rho) - thres < epsi_hat: # Yue: I changed this.
			break
		oldx, oldy, oldlamb = x, y, lamb
	return x


def fixed_point_equation_col(v, l, u, rho, tau):
	tmp = np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)
	z = np.clip(tmp, l, u)
	return tau - rho * np.sum(np.abs(z))

def prox_squared_l1_box_colvec(v, x, l, u, rho, atol=1e-10, rtol=1e-10):
	v, l, u = v - x, l - x, u - x
	if np.any(l > u):
		raise ValueError("Require element-wise l ≤ u.")
	if rho <= 0:
		raise ValueError("rho must be strictly positive.")
    
	low = 0
	high = 1 + rho * np.sum(np.maximum(np.maximum(np.abs(v), np.abs(l)), np.abs(u))) #Yue: I added rho * 
	while high - low > atol:
		tau = 0.5 * (low + high)
		F_tau = fixed_point_equation_col(v, l, u, rho, tau)
		if abs(F_tau) <= rtol:
			break
		if F_tau > 0.0:
			high = tau
		else:
			low = tau

	tmp = np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)
	z = np.clip(tmp, l, u)

	return z + x


rho   = 1.0
R = 20
alpha = 10.0
tol   = 1e-4
max_it = 30000
trials = 10
dims   = [2**k for k in range(6, 14)]
csv_path = "result_box.csv"
def run_one_trial(seed, d):
    np.random.seed(seed)
    v = np.random.normal(size = (d,1))
    w = np.random.normal(size = (d,1))
    l, u = -R*np.ones((d,1)), R*np.ones((d,1))

    # --- exact solver
    t0 = time.perf_counter()
    x_ex = prox_squared_l1_box_colvec(v, w, l, u, rho)
    t_ex = time.perf_counter() - t0
    F_ex = F(x_ex, v, w, rho) # Yue: I changed this.
    beta = 0.1 + 0.3*np.log2(d) # Yue: different betas affect the performance
    # --- ADMM
    t0 = time.perf_counter()
    x_adm = prox_1_admm_box(v, w, rho, R, beta, 1e-12, max_ti= t_ex*100, thres = F_ex)
    t_adm = time.perf_counter() - t0
    F_adm = F(x_adm, v, w, rho) # Yue: I changed this.

    return (t_ex, t_adm, F_ex, F_adm, np.linalg.norm(x_ex-w,1) - alpha,
            np.linalg.norm(x_adm-w,1)-alpha,
            np.linalg.norm(x_ex - x_adm),np.all((x_ex>=l)&(x_ex<=u)), np.all((x_adm>=l)&(x_adm<=u)))             # l2-distance
with open(csv_path, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["d","trial",
                 "time_exact","time_admm",
                 "F_exact","F_admm", "l1gap_exact",
                 "l1gap_admm","l2diff","box_exact","box_admm","mu"])
    for d in dims:
        for t in range(trials):
            res = run_one_trial(seed=123+t, d=d)
            wr.writerow([d, t, *res])

print("Results written to", csv_path)

# ----------------------------------------------------------------
# 5.  read CSV and aggregate
# ----------------------------------------------------------------
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
d_col       = data[:,0]
time_exact  = data[:,2]
time_admm   = data[:,3]
F_exact     = data[:,4]
F_admm      = data[:,5]
mu = data[:, 6]

unique_d = np.unique(d_col)
avg_time_exact = [time_exact[d_col==d].mean() for d in unique_d]
avg_time_admm  = [time_admm [d_col==d].mean() for d in unique_d]
avg_Fgap       = [(F_admm[d_col==d] - F_exact[d_col==d]).mean()
                  for d in unique_d] #Yue: I removed the absolute value sign

# ----------------------------------------------------------------
# 6.  plots
# ----------------------------------------------------------------
plt.figure(figsize=(7,3.2))

# (a) run-time
plt.subplot(1,2,1)
plt.plot(unique_d, avg_time_exact, 'o-', label="exact")
plt.plot(unique_d, avg_time_admm , 's-', label="ADMM")
plt.xscale('log', base=2); plt.yscale('log')
plt.xlabel("dimension d"); plt.ylabel("average run-time (s)")
plt.title("run-time vs. d"); plt.legend()

# (b) objective gap
plt.subplot(1,2,2)
plt.plot(unique_d, avg_Fgap, 'o-')
plt.xscale('log', base=2); plt.yscale('log')
plt.xlabel("dimension d"); plt.ylabel("|F_ADMM − F_exact|")
plt.title("objective gap vs. d")

plt.tight_layout(); plt.show()
plt.savefig("box_admm_compare.pdf")
