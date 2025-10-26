import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams['lines.linewidth'] = 0.8

# 1) Base network
class BaseNet(nn.Module):
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

# 2) Soft PINN model
class SoftPINN1D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.base = BaseNet(layers)
    def forward(self, x):
        return self.base(x)

# 3) Hard PINN model: u = A(x) + M(x)*NN(x)
class HardPINN1D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.base = BaseNet(layers)
    def forward(self, x):
        # A(x) = x^2 - x ensures BCs: A(0)=0, A(1)=0
        A = x**2 - x
        # mask M(x) = x*(x-1) vanishes at 0,1
        M = x*(x-1)
        return A + M * self.base(x)

# PDE residual u_xx - 2 = 0
def pde_residual(model, x):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    return u_xx - 2.0

# exact solution and BCs
def exact_solution(x):
    return x**2 - x

# train function
def train_model(model, N_f=100, beta=1.0, epochs=3000):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    x_f = torch.linspace(0,1,N_f+2)[1:-1].unsqueeze(1)
    x_b = torch.tensor([[0.0],[1.0]])
    u_b = exact_solution(x_b)

    x_eval = torch.linspace(0,1,200).unsqueeze(1)
    u_exact = exact_solution(x_eval).detach()

    loss_hist = []
    error_hist = []

    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        # PDE loss
        res = pde_residual(model, x_f)
        loss_f = (res**2).mean()
        # BC loss
        u_pred_b = model(x_b)
        loss_b = ((u_pred_b - u_b)**2).mean()
        loss = loss_f + beta * loss_b
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        # compute L2 error
        with torch.no_grad():
            u_pred = model(x_eval)
            l2 = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
            error_hist.append(l2.item())

    return loss_hist, error_hist

# initialize and train both models
torch.manual_seed(42)
soft_model = SoftPINN1D([1,50,50,1])
hard_model = HardPINN1D([1,50,50,1])

soft_loss, soft_error = train_model(soft_model)
hard_loss, hard_error = train_model(hard_model)

# Plot comparison
fig, axes = plt.subplots(1,2,figsize=(12,4))

# Loss comparison
axes[0].plot(soft_loss, label='Soft PINN')
axes[0].plot(hard_loss, label='Hard PINN')
axes[0].set_yscale('log')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Loss: Soft vs Hard Constraint')
axes[0].legend(loc='upper right', bbox_to_anchor=(1.3,1))
axes[0].grid(True)

# Error comparison
axes[1].plot(soft_error, label='Soft PINN')
axes[1].plot(hard_error, label='Hard PINN')
axes[1].set_yscale('log')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('L2 Relative Error')
axes[1].set_title('Error: Soft vs Hard Constraint')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
