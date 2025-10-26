import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

plt.rcParams['lines.linewidth'] = 0.8

# PINN model definition
class PINN1D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.act = nn.Tanh()
        self.net = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )

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
    # collocation and boundary data
    x_f = torch.linspace(0,1,N_f+2).unsqueeze(1)
    x_b = torch.tensor([[0.0],[1.0]])
    u_b = exact_solution(x_b)

    # evaluation grid for L2 error
    x_eval = torch.linspace(0,1,200).unsqueeze(1)
    u_exact = exact_solution(x_eval).detach()

    # set up optimizers
    use_hybrid = (opt_name == 'Adam+LBFGS')
    if use_hybrid:
        optimizer1 = optim.Adam(model.parameters(), lr=1e-3)
        optimizer2 = optim.LBFGS(
            model.parameters(),
            max_iter=100,
            tolerance_grad=1e-8,
            line_search_fn='strong_wolfe'
        )
        half = epochs // 2
    else:
        if opt_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        elif opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        elif opt_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        elif opt_name == 'AdamSLOW':
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
        elif opt_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

    def calc_loss():
        res = pde_residual(model, x_f)
        loss_f = (res**2).mean()
        u_pred_b = model(x_b)
        loss_b = ((u_pred_b - u_b)**2).mean()
        return loss_f + beta * loss_b

    def closure():
        optimizer2.zero_grad()
        loss = calc_loss()
        loss.backward()
        return loss

    error_history = []
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

        # record time and error
        t1 = time.perf_counter()
        time_history.append(t1 - t0)
        with torch.no_grad():
            u_pred = model(x_eval)
            l2_rel = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
            error_history.append(l2_rel.item())

    return error_history, time_history

def main():
    torch.manual_seed(0)
    optimizers = ['SGD','Adam','AdamW','AdamSLOW','Adam+LBFGS']
    error_histories = {}
    time_histories = {}

    for opt in optimizers:
        model = PINN1D([1,50,50,1])
        err_hist, t_hist = train_with_optimizer(opt, model)
        error_histories[opt] = err_hist
        time_histories[opt] = t_hist

    epochs = range(1, len(next(iter(error_histories.values())))+1)
    # cumulative time for x-axis
    cum_times = {opt: torch.tensor(t_hist).cumsum(0).tolist() 
                 for opt, t_hist in time_histories.items()}

    # Plot: Error vs Epoch and Error vs Time
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Left: vs epoch
    for opt, err_hist in error_histories.items():
        axes[0].plot(epochs, err_hist, label=opt)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('L2 Relative Error')
    axes[0].set_title('Error vs Epoch')
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    axes[0].grid(True)

    # Right: vs cumulative time
    for opt, err_hist in error_histories.items():
        axes[1].plot(cum_times[opt], err_hist, label=opt)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('L2 Relative Error')
    axes[1].set_title('Error vs Wall-Clock Time')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
