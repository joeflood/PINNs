import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from scipy.linalg import solve_banded

plt.rcParams['lines.linewidth'] = 0.8
torch.manual_seed(42)

# ---------- Finite–difference solver -------------------------------- #
def fd_solution(eps, N=400):
    """Central 2nd-order FD solution of −eps·y'' + y' = 1, y(0)=y(1)=0."""
    h   = 1.0/(N+1)
    x   = np.linspace(0, 1, N+2)

    main  =  2*eps/h**2
    upper = -eps/h**2 + 1/(2*h)
    lower = -eps/h**2 - 1/(2*h)

    ab          = np.zeros((3, N))
    ab[0, 1:]   = upper
    ab[1,  :]   = main
    ab[2, :-1]  = lower

    rhs   = np.ones(N)
    y_int = solve_banded((1,1), ab, rhs)
    y     = np.hstack(([0.], y_int, [0.]))
    return x, y

# ---------- PINN solver (simple fully-connected tanh net) ------------ #
def pinn_solution(eps, layers, N_coll=200, epochs=8000, lr=1e-3, device='cpu'):
    class PINN(nn.Module):
        def __init__(self, dims):
            super().__init__()
            mods=[]
            for i in range(len(dims)-1):
                mods.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2: mods.append(nn.Tanh())
            self.net = nn.Sequential(*mods)
        def forward(self, x): return self.net(x)

    model = PINN(layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    mse   = nn.MSELoss()

    x_c   = torch.rand(N_coll,1, requires_grad=True, device=device)
    x_b   = torch.tensor([[0.],[1.]], requires_grad=True, device=device)
    y_b   = torch.zeros_like(x_b)

    for _ in range(epochs):
        opt.zero_grad()
        y   = model(x_c)
        dy  = torch.autograd.grad(y, x_c, torch.ones_like(y), create_graph=True)[0]
        dyy = torch.autograd.grad(dy, x_c, torch.ones_like(dy), create_graph=True)[0]
        res = -eps*dyy + dy - 1
        loss = mse(res, torch.zeros_like(res)) + mse(model(x_b), y_b)
        loss.backward(); opt.step()
    return model

# -------------------- parameters & run -------------------------------- #
fd_eps  = [5e-3, 1e-3, 5e-4]              # for finite difference
pinn_eps= [5e-2, 3e-2, 1e-2]              # for PINN
layers  = [1, 50, 50, 1]

x_dense  = np.linspace(0,1,500)[:,None]
x_denseT = torch.tensor(x_dense, dtype=torch.float32)

fig, (ax_fd, ax_pinn) = plt.subplots(1,2, figsize=(12,4), sharey=True)
ax_fd.set_title('Finite difference'); ax_pinn.set_title('PINN')
ax_fd.set_xlabel('x'); ax_pinn.set_xlabel('x'); ax_fd.set_ylabel('y(x)')

# --- FD curves ------------------------------------------------------- #
for eps in fd_eps:
    x_fd, y_fd = fd_solution(eps, N=400)
    ax_fd.plot(x_fd, y_fd, label=f'epsilon = {eps:g}')
ax_fd.legend(); ax_fd.grid(True)

# --- PINN curves ----------------------------------------------------- #
for eps in pinn_eps:
    pinn = pinn_solution(eps, layers, N_coll=200, epochs=8000)
    y_p  = pinn(x_denseT).detach().cpu().numpy().ravel()
    ax_pinn.plot(x_dense.ravel(), y_p, label=f'epsilon = {eps:g}')
ax_pinn.legend(); ax_pinn.grid(True)

plt.tight_layout(); plt.show()
