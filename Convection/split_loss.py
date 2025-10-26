import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
torch.manual_seed(42)

# ----------------------------------------------------------------------
# PINN solver – now returns three separate loss curves
# ----------------------------------------------------------------------
def pinn_collocation_solver(eps, layers,
                            N_collocation=200, lr=1e-3,
                            epochs=8000, device='cpu'):

    class PINN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            mods=[]
            for i in range(len(layers)-1):
                mods.append(nn.Linear(layers[i], layers[i+1]))
                if i < len(layers)-2: mods.append(nn.Tanh())
            self.net = nn.Sequential(*mods)
        def forward(self, x): return self.net(x)

    model = PINN(layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    mse   = nn.MSELoss()

    # collocation & BC points
    x_c = torch.rand(N_collocation,1, requires_grad=True, device=device)
    x_b = torch.tensor([[0.0],[1.0]], requires_grad=True, device=device)
    y_b = torch.zeros_like(x_b)

    hist_total, hist_pde, hist_bc = [], [], []

    for epoch in range(epochs):
        opt.zero_grad()
        y   = model(x_c)
        dy  = torch.autograd.grad(y, x_c, torch.ones_like(y), create_graph=True)[0]
        dyy = torch.autograd.grad(dy, x_c, torch.ones_like(dy), create_graph=True)[0]
        res = -eps*dyy + dy - 1
        loss_pde = mse(res, torch.zeros_like(res))
        loss_bc  = mse(model(x_b), y_b)
        loss     = loss_pde + loss_bc

        loss.backward(); opt.step()

        hist_total.append(loss.item())
        hist_pde.append(loss_pde.item())
        hist_bc .append(loss_bc.item())

        if (epoch+1) % 1000 == 0:
            print(f'ε={eps}: epoch {epoch+1}/{epochs}, '
                  f'loss={loss.item():.2e}, '
                  f'PDE={loss_pde.item():.2e}, BC={loss_bc.item():.2e}')

    return model, hist_total, hist_pde, hist_bc

# ----------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------
epsilons       = [5e-2, 3e-2, 1e-2]
layers         = [1, 50, 50, 1]
epochs         = 8000
N_collocation  = 200
lr             = 1e-3

sol_curves = []          # (eps, x_dense, y_pred, y_exact)
loss_curves = []         # (eps, Lpde, Lbc)

for eps in epsilons:
    model, Ltot, Lpde, Lbc = pinn_collocation_solver(
        eps, layers, N_collocation, lr=lr, epochs=epochs)

    y_pred = model(x_dense_t).detach().cpu().numpy().ravel()
    y_ex   = x_dense.ravel() + (np.exp(-(1-x_dense.ravel())/eps) -
                                np.exp(-1/eps)) / (np.exp(-1/eps) - 1)

    sol_curves.append((eps, x_dense.ravel(), y_pred, y_ex))
    loss_curves.append((eps, Lpde, Lbc))

# ------------------------------------------------------------------
# PLOT ALL SOLUTIONS ON ONE GRAPH
# ------------------------------------------------------------------
fig_sol, ax_sol = plt.subplots(figsize=(7,4))
for eps, x_line, y_pin, y_ex in sol_curves:
    ax_sol.plot(x_line, y_pin, label=f'PINN  ε={eps:g}')
    ax_sol.plot(x_line, y_ex , ls='--',  label=f'Exact ε={eps:g}')
ax_sol.set_xlabel('x'); ax_sol.set_ylabel('y(x)')
ax_sol.set_title('PINN vs exact (all ε)')
ax_sol.grid(True); ax_sol.legend()
fig_sol.tight_layout()

# ------------------------------------------------------------------
# PLOT ALL LOSSES ON ONE LOG GRAPH
# ------------------------------------------------------------------
fig_loss, ax_loss = plt.subplots(figsize=(7,4))
epochs_range = np.arange(1, epochs+1)
for eps, Lpde, Lbc in loss_curves:
    ax_loss.plot(epochs_range, Lpde, label=f'PDE ε={eps:g}')
    ax_loss.plot(epochs_range, Lbc , ls=':', label=f'BC  ε={eps:g}')
ax_loss.set_yscale('log')
ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('MSE loss')
ax_loss.set_title('Training loss (solid=PDE, dotted=BC)')
ax_loss.grid(True); ax_loss.legend(ncol=2, fontsize=8)
fig_loss.tight_layout()

plt.show()
