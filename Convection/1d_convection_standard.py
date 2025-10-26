import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

def pinn_collocation_solver(eps, layers, N_collocation=200, N_boundary=2, lr=1e-3, epochs=8000, device='cpu'):
    # Define the PINN
    class PINN(nn.Module):
        def __init__(self, layers):
            super(PINN, self).__init__()
            layer_list = []
            for i in range(len(layers)-1):
                layer_list.append(nn.Linear(layers[i], layers[i+1]))
                if i < len(layers)-2:
                    layer_list.append(nn.Tanh())
            self.net = nn.Sequential(*layer_list)

        def forward(self, x):
            return self.net(x)

    # Initialize network
    model = PINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # Prepare training data
    x_coll = torch.rand(N_collocation, 1, device=device, requires_grad=True)
    x_b = torch.tensor([[0.0],[1.0]], device=device, requires_grad=True)
    y_b = torch.zeros_like(x_b)

    loss_history = []
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        # PDE residual at collocation points
        y_coll = model(x_coll)
        y_coll_x = torch.autograd.grad(y_coll, x_coll, grad_outputs=torch.ones_like(y_coll), create_graph=True)[0]
        y_coll_xx = torch.autograd.grad(y_coll_x, x_coll, grad_outputs=torch.ones_like(y_coll_x), create_graph=True)[0]
        f_coll = -eps * y_coll_xx + y_coll_x - 1.0
        loss_pde = mse_loss(f_coll, torch.zeros_like(f_coll))

        # Boundary loss
        y_b_pred = model(x_b)
        loss_bc = mse_loss(y_b_pred, y_b)

        # Total loss
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch+1) % 1000 == 0:
            print(f"eps={eps}: Epoch {epoch+1}/{epochs}, Loss: {loss.item():.3e}")

    return model, loss_history

if __name__ == "__main__":
    # Epsilons to test
    epsilons = [5e-2, 3e-2, 1e-2]
    layers = [1, 50, 50, 1]
    epochs = 8000
    N_collocation = 200
    lr = 1e-3

    # Prepare for plotting solutions
    x = np.linspace(0,1,200)[:,None]
    x_tensor = torch.tensor(x, dtype=torch.float32)

    fig1, axes = plt.subplots(1,3, figsize=(18,4))
    fig2, ax2 = plt.subplots(figsize=(6,4))

    # Loop over each epsilon
    for i, eps in enumerate(epsilons):
        model, loss_hist = pinn_collocation_solver(eps, layers, N_collocation, epochs=epochs, lr=lr)
        # Predict solution
        y_pred = model(x_tensor).detach().numpy()
        # Exact solution
        y_exact = x + (np.exp(-(1-x)/eps) - np.exp(-1/eps)) / (np.exp(-1/eps) - 1)

        # Plot solution
        ax = axes[i]
        ax.plot(x, y_exact, 'k--', label='Exact')
        ax.plot(x, y_pred, label='PINN')
        ax.set_title(f'eps = {eps}')
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.grid(True)  # Add grid to solution plots
        ax.legend()

        # Plot loss
        ax2.plot(np.arange(1, epochs+1), loss_hist, label=f'eps={eps}')

    fig2.set_tight_layout(True)
    ax2.set_yscale('log')  # Log scale for loss
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss vs Epoch')
    ax2.legend()
    ax2.grid(True)  # Add grid to loss plot
    plt.show()
