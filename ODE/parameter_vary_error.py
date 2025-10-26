import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


plt.rcParams['lines.linewidth'] = 0.8


# PINN model definition (unchanged)
class PINN1D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.act = nn.Tanh()
        net = []
        for i in range(len(layers)-1):
            net.append(nn.Linear(layers[i], layers[i+1]))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        a = x
        for layer in self.net[:-1]:
            a = self.act(layer(a))
        return self.net[-1](a)

# PDE residual (unchanged)
def pde_residual(model, x):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx - 2.0

# exact solution
def exact_solution(x):
    return x**2 - x

# modified training routine that also tracks L2 relative error
def train_pinn(width, depth, beta, N_f, epochs=5000):
    # build model & optimizer
    layers = [1] + [width]*depth + [1]
    model = PINN1D(layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # collocation & BC data
    x_f = torch.linspace(0,1,N_f+2)[1:-1].unsqueeze(1)
    x_b = torch.tensor([[0.0],[1.0]])
    u_b = exact_solution(x_b)

    # fixed evaluation grid (for error)
    x_eval = torch.linspace(0,1,200).unsqueeze(1)
    u_exact = exact_solution(x_eval).detach()

    loss_history  = []
    error_history = []

    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        res    = pde_residual(model, x_f)
        loss_f = (res**2).mean()
        u_bnd  = model(x_b)
        loss_b = ((u_bnd - u_b)**2).mean()
        loss   = loss_f + beta * loss_b
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # compute L2 relative error
        with torch.no_grad():
            u_pred = model(x_eval)
            l2_rel = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
            error_history.append(l2_rel.item())

    return loss_history, error_history

# hyperparameters to sweep
widths = [5, 10, 20, 50]
depths = [1, 2, 4, 8]
betas  = [0.1, 1.0, 10.0]
N_fs   = [1, 5, 10, 50]
epochs = 10000

# collect histories of relative error
histories = {'width':{}, 'depth':{}, 'beta':{}, 'N_f':{}}

# sweep width
for w in widths:
    _, err = train_pinn(width=w, depth=3, beta=1.0, N_f=5, epochs=epochs)
    histories['width'][f'w={w}'] = err

# sweep depth
for d in depths:
    _, err = train_pinn(width=50, depth=d, beta=1.0, N_f=5, epochs=epochs)
    histories['depth'][f'd={d}'] = err

# sweep beta
for b in betas:
    _, err = train_pinn(width=50, depth=3, beta=b, N_f=5, epochs=epochs)
    histories['beta'][f'beta={b}'] = err

# sweep N_f
for nf in N_fs:
    _, err = train_pinn(width=50, depth=3, beta=1.0, N_f=nf, epochs=epochs)
    histories['N_f'][f'Nf={nf}'] = err

# plotting L2 relative error vs epoch for each sweep
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for ax, key in zip(axes, ['width','depth','beta','N_f']):
    for label, err_hist in histories[key].items():
        ax.plot(range(1, epochs+1), err_hist, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L2 Rel Error')
    ax.set_title(f'Relative Error vs Epoch (vary {key})')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1))
    ax.grid(True)

plt.tight_layout()
plt.show()
