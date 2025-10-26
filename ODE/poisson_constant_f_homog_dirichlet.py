import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scienceplots




# 1) Define the PINN
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
        for l in self.net[:-1]:
            a = self.act(l(a))
        return self.net[-1](a)

# 2) PDE residual u_xx - 2 = 0
def pde_residual(model, x):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    return u_xx - 2.0

# 3) Exact solution and BCs
def exact_solution(x):
    return x**2 - x

# 4) Data: interior collocation + Dirichlet BC
def generate_data(N_f=1):
    x_f = torch.rand(N_f,1)                  # interior points
    x_b = torch.tensor([[0.0],[1.0]])        # boundary points
    u_b = exact_solution(x_b)
    return x_f, x_b, u_b

# 5) Training
def train(model, optimizer, epochs=3000):
    l2_error_history = []
    x_eval = torch.linspace(0, 1, 200).unsqueeze(1)
    u_exact_eval = exact_solution(x_eval).detach()

    x_f, x_b, u_b = generate_data()
    loss_hist = []
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        res = pde_residual(model, x_f)
        loss_f = (res**2).mean()
        u_pred_b = model(x_b)
        loss_b = ((u_pred_b - u_b)**2).mean()
        loss = loss_f + loss_b
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        if epoch % 500 == 0:
            print(f"[{epoch}/{epochs}]  Loss: {loss.item():.2e}")
        with torch.no_grad():
            u_pred_eval = model(x_eval)
            l2 = torch.norm(u_pred_eval - u_exact_eval) / torch.norm(u_exact_eval)
            l2_error_history.append(l2.item())

    return loss_hist, l2_error_history

# 6) Main
if __name__ == "__main__":
    torch.manual_seed(0)
    model = PINN1D([1, 50, 50, 50, 1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_history, l2_error_history = train(model, optimizer, epochs=600)

    # Test grid
    x_test = torch.linspace(0,1,200).unsqueeze(1)
    # 1) Compute predictions & exact solution without gradients, and detach to save memory
    with torch.no_grad():
        u_pred  = model(x_test).detach()
        u_exact = exact_solution(x_test).detach()

    # 2) Compute residual with gradients enabled
    residual_test = pde_residual(model, x_test).detach()

    # Create a single figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot 1: PINN vs Exact
    axes[0].plot(x_test.numpy(), u_pred.numpy())
    axes[0].plot(x_test.numpy(), u_exact.numpy())
    axes[0].legend(['PINN', 'Exact'])
    axes[0].set_title('Solution: PINN vs Exact')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].grid(True)

    # Plot 2: Loss vs Epoch
    axes[1].plot(loss_history)
    axes[1].set_yscale('log')
    axes[1].set_title('Training Loss vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    # Plot 3: PDE Residual
    axes[2].plot(x_test.numpy(), residual_test.numpy())
    axes[2].set_title('PDE Residual $u_{xx}-2$ across [0,1]')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Residual')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(l2_error_history)
    plt.yscale('log')
    plt.title('L2 Relative Error vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

