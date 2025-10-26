import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

def shishkin_1d(N, eps, alpha=1.0, return_full=False):
    assert N % 2 == 0, "N must be even for Shishkin mesh"
    sigma = min(0.5, alpha * eps * np.log(N))
    N2 = N // 2
    h_c = sigma / N2
    h_f = (1.0 - sigma) / N2
    x = np.zeros(N+1)
    x[:N2+1]       = np.arange(N2+1)        * h_c
    x[N2+1:]       = sigma + np.arange(1,N2+1) * h_f
    if return_full:
        return x, sigma, N2, h_c, h_f
    return x[1:-1].reshape(-1,1)           # interior only

def describe_shishkin(N, eps, alpha=1.0):
    x_full, sigma, N2, h_c, h_f = shishkin_1d(N, eps, alpha, return_full=True)
    n_coarse = N2-1                          # interior coarse pts
    n_fine   = N2-1                          # interior fine  pts
    dens_c   = n_coarse / sigma
    dens_f   = n_fine   / (1-sigma)
    print(f"σ = {sigma:.4f},  h_c = {h_c:.3e},  h_f = {h_f:.3e}")
    print(f"coarse pts: {n_coarse}  (density {dens_c:.1f}/unit)")
    print(f"fine   pts: {n_fine}    (density {dens_f:.1f}/unit)")
    return x_full[1:-1].reshape(-1,1)


def pinn_collocation_solver(eps, layers, N_collocation=200, N_boundary=2,
                             lr=1e-3, epochs=5000, mesh_type='uniform',
                             shishkin_alpha=1.0, device='cpu'):
    # Define PINN
    class PINN(nn.Module):
        def __init__(self, layers):
            super(PINN, self).__init__()
            layer_list = []
            for i in range(len(layers)-1):
                layer_list.append(nn.Linear(layers[i], layers[i+1]))
                if i < len(layers)-2:
                    layer_list.append(nn.Tanh())
            self.net = nn.Sequential(*layer_list)
        def forward(self, x): return self.net(x)

    model = PINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # Collocation points
    if mesh_type == 'shishkin':
        x_coll_np = shishkin_1d(N_collocation, eps, alpha=shishkin_alpha)
    else:
        xi = np.linspace(0, 1, N_collocation + 2)[1:-1]
        x_coll_np = xi.reshape(-1,1)
    x_coll = torch.tensor(x_coll_np, dtype=torch.float32,
                          device=device, requires_grad=True)
    # Boundary points
    x_b = torch.tensor([[0.0],[1.0]], device=device, requires_grad=True)
    y_b = torch.zeros_like(x_b)

    loss_history = []
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        y_coll = model(x_coll)
        y_coll_x = torch.autograd.grad(y_coll, x_coll,
            grad_outputs=torch.ones_like(y_coll), create_graph=True)[0]
        y_coll_xx = torch.autograd.grad(y_coll_x, x_coll,
            grad_outputs=torch.ones_like(y_coll_x), create_graph=True)[0]
        f_coll = -eps * y_coll_xx + y_coll_x - 1.0
        loss_pde = mse_loss(f_coll, torch.zeros_like(f_coll))

        y_b_pred = model(x_b)
        loss_bc = mse_loss(y_b_pred, y_b)
        loss = loss_pde + 10*loss_bc
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history

if __name__ == "__main__":
    # Parameters
    eps = 3e-3
    layers = [1,50,50,1]
    epochs = 10000
    N_collocation = 200
    lr = 1e-3

    # 1) Compare solution and loss for uniform vs standard Shishkin
    loss_dict = {}
    models = {}
    for mesh in ['uniform', 'shishkin']:
        model, loss_hist = pinn_collocation_solver(
            eps=eps, layers=layers, N_collocation=N_collocation,
            epochs=epochs, lr=lr, mesh_type=mesh, shishkin_alpha=1.0)
        loss_dict[mesh] = loss_hist
        models[mesh] = model

    x = np.linspace(0,1,200)[:,None]
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Plot solutions
    plt.figure(figsize=(10,4))
    for mesh, model in models.items():
        y_pred = model(x_tensor).detach().numpy()
        plt.plot(x, y_pred, '--', label=f'PINN ({mesh})')
    # exact
    y_exact = x + (np.exp(-(1-x)/eps) - np.exp(-1/eps)) / (np.exp(-1/eps) - 1)
    plt.plot(x, y_exact, 'k-', label='Exact')
    plt.xlabel('x'); plt.ylabel('y(x)'); plt.title('PINN vs Exact Solutions')
    plt.legend(); plt.grid(True);
    plt.tight_layout()
    plt.show()

    # Plot loss comparison
    plt.figure(figsize=(8,4))
    epochs_range = np.arange(1, epochs+1)
    for mesh, loss_hist in loss_dict.items():
        plt.plot(epochs_range, loss_hist, label=f'{mesh.capitalize()} Mesh')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Training Loss vs Epoch (eps={eps})')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # 2) Parameter sweep: vary Shishkin alpha
    alphas = [1.0, 1.5, 2.0, 5.0]
    sweep_losses = {}
# α-sweep loop (two returns)
    for alpha in alphas:
        model, loss_hist = pinn_collocation_solver(
            eps, layers, N_collocation, epochs, lr,
            mesh_type='shishkin', shishkin_alpha=alpha)
        describe_shishkin(N_collocation, eps, alpha)
        sweep_losses[alpha] = loss_hist


    # Plot sweep loss histories
    plt.figure(figsize=(8,5))
    for alpha, loss_hist in sweep_losses.items():
        plt.plot(epochs_range, loss_hist, label=f'alpha={alpha}')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss vs Epoch: Shishkin-ness Parameter Sweep')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
