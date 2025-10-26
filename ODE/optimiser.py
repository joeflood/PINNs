import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np

plt.rcParams['lines.linewidth'] = 0.8

# simple 1D PINN
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

def exact_solution(x):
    return x**2 - x

def pde_residual(model, x):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx - 2.0

def train_with_optimizer(opt_name, model, N_f=5, beta=1.0, epochs=5000):
    # data
    x_f = torch.linspace(0,1,N_f+2)[1:-1].unsqueeze(1)
    x_b = torch.tensor([[0.0],[1.0]])
    u_b = exact_solution(x_b)

    # define loss computation
    def calc_loss():
        res = pde_residual(model, x_f)
        loss_f = (res**2).mean()
        u_pred_b = model(x_b)
        loss_b = ((u_pred_b - u_b)**2).mean()
        return loss_f + beta * loss_b

    # choose optimizers
    use_hybrid = (opt_name == 'Adam+LBFGS')
    if use_hybrid:
        optimizer1 = optim.Adam(model.parameters(), lr=1e-3)
        optimizer2 = optim.LBFGS(
            model.parameters(),
            max_iter=500,
            tolerance_grad=1e-15,
            line_search_fn='strong_wolfe'
        )
        half = epochs // 2
    else:
        if opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        elif opt_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        elif opt_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        elif opt_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
        elif opt_name == 'AdamSLOW':
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

    # closure for L-BFGS
    def closure():
        optimizer2.zero_grad()
        loss = calc_loss()
        loss.backward()
        return loss

    loss_history = []
    time_history = []

    for epoch in range(1, epochs+1):
        t0 = time.perf_counter()

        if use_hybrid and epoch <= half:
            optimizer1.zero_grad()
            loss = calc_loss()
            loss.backward()
            optimizer1.step()
        elif use_hybrid:
            loss = optimizer2.step(closure)
        else:
            optimizer.zero_grad()
            loss = calc_loss()
            loss.backward()
            optimizer.step()

        t1 = time.perf_counter()
        loss_history.append(loss.item())
        time_history.append(t1 - t0)

    return loss_history, time_history

def main():
    torch.manual_seed(0)
    optimizers = ['SGD','Adam','AdamW','AdamSLOW','Adam+LBFGS']
    histories_loss = {}
    histories_time = {}

    for opt in optimizers:
        model = PINN1D([1,50,50,1])
        loss_hist, time_hist = train_with_optimizer(opt, model)
        histories_loss[opt] = loss_hist
        histories_time[opt] = np.cumsum(time_hist)  # cumulative time

    epochs = range(1, len(next(iter(histories_loss.values())))+1)

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Loss vs Epoch
    for opt, loss_hist in histories_loss.items():
        axes[0].plot(epochs, loss_hist, label=opt)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Loss vs Epoch')
    axes[0].grid(True)

    # Loss vs Time
    for opt, loss_hist in histories_loss.items():
        axes[1].plot(histories_time[opt], loss_hist, label=opt)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Loss vs Wall-Clock Time')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
