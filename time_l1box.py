import numpy as np, time, csv, datetime, argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt

def F(x, v, rho):
    return 0.5*np.linalg.norm(x - v, 2)**2 + 0.5*rho*np.linalg.norm(x, 1)**2


def x_coord(v_i, w_i, tau, mu, l_i, u_i):
    if w_i == 0:
        tmp = np.sign(v_i) * np.maximum(np.abs(v_i) - tau - mu, 0.0)
        x   = np.clip(tmp, l_i, u_i)
    elif tau < -v_i - mu + min(0.0, w_i):
        x =  tau + v_i + mu
    elif tau <  v_i - mu - max(0.0, w_i):
        x = -tau + v_i - mu
    elif (tau <= v_i*np.sign(w_i) + mu and
          tau >= v_i*np.sign(w_i) + mu - abs(w_i)):
        x = -tau*np.sign(w_i) + v_i + mu*np.sign(w_i)
    elif tau >= abs(v_i + mu*np.sign(w_i)):
        x = 0.0
    else:
        x = w_i
    return min(max(x, l_i), u_i)


def x_vec(v, w, tau, mu, l, u): 
    v, w, l, u = map(np.ravel, (v, w, l, u))
    return np.array([x_coord(v[i], w[i], tau, mu, l[i], u[i])
                     for i in range(v.size)])

def solve_tau(v, w, mu, rho, l, u, tol=1e-8):
    if mu == 0.0:
       def F(tau):
           return tau - rho * np.sum(np.abs(np.clip(np.sign(v) * np.maximum(np.abs(v) - tau, 0.0), l, u)))
       lo, hi = 0.0, rho*np.sum(np.maximum(np.abs(l), np.abs(u)))
       while hi - lo > tol:
           mid = 0.5*(lo+hi)
           (hi, lo) = (mid, lo) if F(mid) > 0 else (hi, mid)
       return 0.5*(lo+hi)
    def f(tau):
        return tau - rho * np.sum(np.abs(x_vec(v, w, tau, mu, l, u)))
    lo, hi = 0.0, rho*np.sum(np.maximum(np.abs(l), np.abs(u)))
    while f(hi) < 0: hi *= 2.0
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        (hi, lo) = (mid, lo) if f(mid) > 0 else (hi, mid)
    return 0.5*(lo+hi)

def prox_sqL1_box(v, rho, w, alpha, l, u,
                  tau_tol=1e-8, mu_tol=1e-6, max_iter=100, tolc = 1e-12):
    v, w, l, u = map(lambda a: np.asarray(a,float).ravel(), (v,w,l,u))

    mu_lo = 0.0
    tau0  = solve_tau(v, w, mu_lo, rho, l, u, tau_tol)
    x = np.clip(np.sign(v) * np.maximum(np.abs(v) - tau0, 0.0), l, u)
    if np.linalg.norm(x-w,1) <= alpha: return x, 0

    mu_hi = np.max(np.abs(np.clip(w,l,u) - v)) + rho*np.linalg.norm(np.clip(w,l,u),1)

    while mu_hi - mu_lo > mu_tol or np.linalg.norm(x-w,1) > alpha + tolc:
        mu   = 0.5*(mu_lo + mu_hi)
        tau  = solve_tau(v, w, mu, rho, l, u, tau_tol)
        x    = x_vec(v, w, tau, mu, l, u)
        gap  = np.linalg.norm(x-w,1) - alpha
        (mu_lo, mu_hi) = (mu, mu_hi) if gap > 0 else (mu_lo, mu)
    return x, mu

def prox_sqL1_unconstrained(q, rho_t, tol=1e-6):
    if rho_t == 0:
        return q.copy()
    lo, hi = 0.0, np.max(np.abs(q))
    while hi - lo > tol:
        lam = 0.5*(lo + hi)
        s   = np.sum(np.maximum(np.abs(q) - lam, 0.0))
        phi = lam - rho_t * s
        if phi > 0:
            hi = lam
        else:
            lo = lam
    lam = 0.5*(lo + hi)
    return np.sign(q) * np.maximum(np.abs(q) - lam, 0.0)

def proj_l1_ball(v, alpha):
    if np.linalg.norm(v,1) <= alpha: return v
    u = np.sort(np.abs(v))[::-1]; cssv = np.cumsum(u)
    k = np.nonzero(u*np.arange(1,len(u)+1) > cssv - alpha)[0][-1]
    theta = (cssv[k]-alpha)/(k+1.0)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


def admm_sqL1_box(v, rho, w, alpha, l, u,
                  beta=1.0, max_iter=100000, tol=1e-6, log_every=100, thres = 0, max_ti = 0, tolc = 1e-12):
    time_start = time.time()
    v, w, l, u = map(lambda a: np.asarray(a,float).ravel(), (v,w,l,u))
    x = np.clip(v, l, u)
    z = x.copy()
    y = proj_l1_ball(x-w, alpha)
    u1 = np.zeros_like(x)
    u2 = np.zeros_like(x)

    while True:
        now = time.time()
        if now - time_start > max_ti:
            break
        q = (v + beta*(z-u1) + beta*(w+y-u2)) / (1 + 2*beta)
        x = prox_sqL1_unconstrained(q, rho/(1+2*beta))

        z_old = z.copy()
        z = np.clip(x + u1, l, u)

        y_old = y.copy()
        y = proj_l1_ball(x - w + u2, alpha)

        u1 += (x - z)
        u2 += (x - w - y)

        r = max(np.linalg.norm(x-z,np.inf),
                np.linalg.norm(x-w-y,np.inf))
        s = beta*max(np.linalg.norm(z-z_old,np.inf),
                     np.linalg.norm(y-y_old,np.inf))
        p = np.linalg.norm(y, 1) - alpha
        if (np.linalg.norm(z - w, 1) < alpha + tolc and F(z,v,rho) < thres):
            break

    return z



rho   = 1.0
alpha = 10.0
tol   = 1e-4
max_it = 30000
trials = 10
dims   = [2**k for k in range(6, 14)]
csv_path = "result_sqL1.csv"
def run_one_trial(seed, d):
    np.random.seed(seed)
    v = np.random.uniform(-50,50,d)
    w = np.random.uniform(-20,20,d)
    l, u = -20*np.ones(d), 20*np.ones(d)

    # --- our solver
    t0 = time.perf_counter()
    x_ex, mu = prox_sqL1_box(v, rho, w, alpha, l, u)
    t_ex = time.perf_counter() - t0
    F_ex = F(x_ex, v, rho)
    beta = 100+300*np.log2(d)
    # --- ADMM
    t0 = time.perf_counter()
    x_adm = admm_sqL1_box(v, rho, w, alpha, l, u,
                          beta=beta, log_every=0, max_iter=max_it, thres=F_ex, max_ti = t_ex*10)
    t_adm = time.perf_counter() - t0
    F_adm = F(x_adm, v, rho)

    return (t_ex, t_adm, F_ex, F_adm, np.linalg.norm(x_ex-w,1) - alpha,
            np.linalg.norm(x_adm-w,1)-alpha,
            np.linalg.norm(x_ex - x_adm),np.all((x_ex>=l)&(x_ex<=u)), np.all((x_adm>=l)&(x_adm<=u)), mu)
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
avg_Fgap       = [np.abs(F_admm[d_col==d] - F_exact[d_col==d]).mean()
                  for d in unique_d]

plt.figure(figsize=(7,3.2))

plt.subplot(1,2,1)
plt.plot(unique_d, avg_time_exact, 'o-', label="exact")
plt.plot(unique_d, avg_time_admm , 's-', label="ADMM")
plt.xscale('log', base=2); plt.yscale('log')
plt.xlabel("dimension d"); plt.ylabel("average run-time (s)")
plt.title("run-time vs. d"); plt.legend()

plt.subplot(1,2,2)
plt.plot(unique_d, avg_Fgap, 'o-')
plt.xscale('log', base=2); plt.yscale('log')
plt.xlabel("dimension d"); plt.ylabel("|F_ADMM − F_exact|")
plt.title("objective gap vs. d")

plt.tight_layout(); plt.show()
plt.savefig("sqL1_box_admm_compare.pdf")
