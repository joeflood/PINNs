import argparse
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
def generate_data(N_f=5):
    x_f = torch.linspace(0, 1, N_f+2)[1:-1].unsqueeze(1)
    x_b = torch.tensor([[0.0],[1.0]])        # boundary points
    u_b = exact_solution(x_b)
    return x_f, x_b, u_b

# 5) Training (returns histories and snapshots)
def train(model, optimizer, epochs=3000, checkpoints=None):
    # prepare eval grid and exact
    x_eval = torch.linspace(0, 1, 200).unsqueeze(1)
    u_exact_eval = exact_solution(x_eval).detach()

    x_f, x_b, u_b = generate_data(N_f=100)
    loss_history = []
    l2_error_history = []
    pde_residual_history = []
    u_snapshots = {}

    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        # PDE loss
        res = pde_residual(model, x_f)
        loss_f = (res**2).mean()
        pde_residual_history.append(torch.sqrt(loss_f).item())

        # BC loss
        u_pred_b = model(x_b)
        loss_b = ((u_pred_b - u_b)**2).mean()

        # total loss
        loss = loss_f + loss_b
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # L2 relative error on eval grid
        with torch.no_grad():
            u_pred_eval = model(x_eval)
            l2 = torch.norm(u_pred_eval - u_exact_eval) / torch.norm(u_exact_eval)
            l2_error_history.append(l2.item())
            # snapshot
            if checkpoints and epoch in checkpoints:
                u_snapshots[epoch] = u_pred_eval.numpy()

        if epoch % 500 == 0:
            print(f"[{epoch}/{epochs}]  Loss: {loss.item():.2e}")

    return loss_history, l2_error_history, pde_residual_history, u_snapshots

# 6) Main
if __name__ == "__main__":
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="If set, retrain the model; otherwise load existing.")
    args = parser.parse_args()

    # specify epochs at which to snapshot
    checkpoints = [0, 1, 5, 10, 50, 100, 500, 1000, 5000]

    model = PINN1D([1, 50, 50, 50, 1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if args.retrain:
        # retrain and save everything
        loss_history, l2_error_history, pde_residual_history, snapshots = train(
            model, optimizer, epochs=5000, checkpoints=checkpoints
        )
        torch.save(model.state_dict(), "pinn_model.pth")
        torch.save(loss_history, "loss_history.pt")
        torch.save(l2_error_history, "l2_error_history.pt")
        torch.save(pde_residual_history, "pde_residual_history.pt")
        torch.save(snapshots, "snapshots.pt")
    else:
        # load pretrained model + histories + snapshots
        model.load_state_dict(torch.load("pinn_model.pth", map_location="cpu"))
        loss_history = torch.load("loss_history.pt")
        l2_error_history = torch.load("l2_error_history.pt")
        pde_residual_history = torch.load("pde_residual_history.pt")
        snapshots = torch.load("snapshots.pt")

    # Prepare test grid
    x_test = torch.linspace(0, 1, 200).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(x_test).numpy()
        u_exact = exact_solution(x_test).numpy()
    residual_test = pde_residual(model, x_test).detach().numpy()

    # Additional plot: snapshots of solution at checkpoints
    plt.figure(figsize=(7,5))
    for epoch, u_vals in snapshots.items():
        plt.plot(x_test.numpy(), u_vals, label=f'Epoch {epoch}')
    plt.plot(x_test.numpy(), u_exact, 'k--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.ylim([-0.6, 0.2])
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    # Make room on the right

    plt.subplots_adjust(right=0.75)
    plt.show()
