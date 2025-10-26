import time                                    #  ← NEW
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

plt.rcParams['lines.linewidth'] = 0.8

# ------------------------------------------------------------------ #
def fd_solution(eps, N=200):
    """Solve  −eps y'' + y' = 1,  y(0)=y(1)=0  on a uniform grid."""
    h   = 1.0/(N+1)
    x   = np.linspace(0, 1, N+2)

    main  =  2*eps/h**2
    upper = -eps/h**2 + 1/(2*h)
    lower = -eps/h**2 - 1/(2*h)

    ab            = np.zeros((3, N))
    ab[0, 1:]     = upper
    ab[1, :]      = main
    ab[2, :-1]    = lower
    rhs = np.ones(N)

    y_int = solve_banded((1, 1), ab, rhs)
    y     = np.hstack(([0.], y_int, [0.]))
    return x, y
# ------------------------------------------------------------------ #

eps_values  = [5e-3, 3e-3, 1e-3]
fig, axes   = plt.subplots(1, 3, figsize=(15,4), sharey=True)
times_ms    = []                            # store elapsed times

for ax, eps in zip(axes, eps_values):
    t0 = time.perf_counter()                # ---------- start timer
    x_fd, y_fd = fd_solution(eps, N=200)
    elapsed = (time.perf_counter() - t0)*1e3   # milliseconds
    times_ms.append(elapsed)

    # FD curve
    ax.plot(x_fd, y_fd, label='FD')

    # Exact solution
    x_ex  = np.linspace(0,1,600)
    y_ex  = x_ex + (np.exp(-(1-x_ex)/eps) - np.exp(-1/eps)) / (np.exp(-1/eps) - 1)
    ax.plot(x_ex, y_ex, 'k--', label='Exact')

    # Annotate time
    ax.text(0.05, 0.05, f'{elapsed:.2f} ms', transform=ax.transAxes,
            fontsize=8, bbox=dict(boxstyle='round', fc='w', ec='0.5'))

    # Cosmetics
    ax.set_title(rf'$\varepsilon = {eps}$')
    ax.set_xlabel('x')
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel('y(x)')
fig.suptitle('Finite Difference vs Exact (timings inside panels)')
fig.tight_layout()
plt.show()

# (Optional) print summary to console
for eps, t in zip(eps_values, times_ms):
    print(f'ε = {eps:6g}  →  {t:8.2f} ms')
