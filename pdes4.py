import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_units, activation='tanh'):
        super(FeedforwardNN, self).__init__()
        
        # Activation function lookup
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'sine': lambda: torch.sin,  # Custom sine activation
            'gelu': nn.GELU,
            'swish': nn.SiLU  # Swish is also called SiLU in PyTorch
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation function: {activation}")
        self.activation_fn = activations[activation]()

        # Construct layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(self.activation_fn)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self.activation_fn)
        layers.append(nn.Linear(hidden_units, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Forcing function: f(x, y) = 13pi^2 sin(2pix) sin(3piy)
# def f_rhs(x, y):
#     return 13 * np.pi**2 * torch.sin(2 * np.pi * x) * torch.sin(3 * np.pi * y)

# Function to sample points in the interior and on the boundary
def sample_points(n_interior=1000, n_boundary=200):
    # Interior: random points in (0,1)×(0,1)
    x_interior = torch.rand((n_interior, 2), dtype=torch.float32)

    # Boundary: concatenate edges of square
    edge = torch.linspace(0, 1, n_boundary // 4)
    xb = torch.cat([
        torch.stack([torch.zeros_like(edge), edge], dim=1),
        torch.stack([torch.ones_like(edge), edge], dim=1),   
        torch.stack([edge, torch.zeros_like(edge)], dim=1),
        torch.stack([edge, torch.zeros_like(edge)], dim=1),

    ])
    return x_interior, xb

# Loss function
def poisson_loss(model, x_interior, x_boundary, gamma=100.0):
    x_interior.requires_grad_(True)
    u = model(x_interior)

    # Compute Laplacian using autograd
    grad_u = torch.autograd.grad(u, x_interior, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    d2u_dx2 = torch.autograd.grad(du_dx, x_interior, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(du_dy, x_interior, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 1:2]
    lap_u = d2u_dx2 - d2u_dy2

    # PDE residual
    # f_vals = f_rhs(x_interior[:, 0:1], x_interior[:, 1:2])
    # loss_pde = torch.mean((lap_u + f_vals)**2)
    loss_pde = torch.mean((lap_u)**2)

    # Boundary loss
    x_boundary.requires_grad_(True)
    u_boundary = model(x_boundary)
    grad_u = torch.autograd.grad(u_boundary, x_boundary, grad_outputs=torch.ones_like(u_boundary), create_graph=True)[0]
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    
    loss_bc1 = torch.mean(u_boundary[0:100]**2)
    loss_bc2 = torch.mean((torch.sin(np.pi * x_boundary[:,0:1][100:150]) - u_boundary[100:150])**2)
    loss_bc3 = torch.mean((-np.pi * torch.sin(np.pi * x_boundary[:,0:1][150:200]) - du_dy[150:200])**2)

    loss_bc = loss_bc1 + loss_bc2 + loss_bc3

    return loss_pde + gamma * loss_bc

# Model training function
def train_poisson(model, epochs=3000, lr=1e-3, gamma=100.0, verbose=True):
    x_int, x_bd = sample_points()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    store_loss=[]
    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = poisson_loss(model, x_int, x_bd, gamma)
        store_loss.append(loss.item())
        loss.backward()
        optimiser.step()

        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6e}")
    return store_loss

# Sensecheck the model has worked
max_epochs=10000
model = FeedforwardNN(input_dim=2, output_dim=1, hidden_layers=3, hidden_units=30, activation='swish')
store_loss = train_poisson(model, epochs=max_epochs, lr=1e-3, gamma=100.0)

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32)

with torch.no_grad():
    u_pred = model(xy).view(100, 100).numpy()

## The analytical solution
# def g_analytic(point):
#     x, t = point
#     return np.sin(np.pi * x) * np.cos(np.pi * t) - np.sin(np.pi * x) * np.sin(np.pi * t)
u_true = np.sin(np.pi * X) * np.cos(np.pi * Y) - np.sin(np.pi * X) * np.sin(np.pi * Y)

# Ensure they're ont the same scale
# Compute global color limits
vmin = min(u_pred.min(), u_true.min())
vmax = max(u_pred.max(), u_true.max())
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Plot NN prediction
cf1 = axes[0].contourf(X, Y, u_pred, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title("NN Prediction")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")

# Plot exact solution
cf2 = axes[1].contourf(X, Y, u_true, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title("Exact Solution")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")

# Add single shared colorbar
cbar = fig.colorbar(cf2, ax=axes, orientation='vertical', shrink=0.9, aspect=30, pad=0.02)
cbar.set_label("Solution value", rotation=270, labelpad=15)
abs_error = np.abs(u_pred - u_true)

# plt.suptitle("Neural Network vs Exact Solution", fontsize=14)
plt.show()


fig2 = plt.figure(figsize=(15, 5))

ax1=plt.subplot(1, 3, 1)
plt.contourf(X, Y, u_pred, levels=100, cmap="viridis")
plt.colorbar(label="Predicted $u_p(x,y)$")
plt.title("Predicted Solution")

ax2=plt.subplot(1, 3, 2)
plt.contourf(X, Y, u_true, levels=100, cmap="viridis")
plt.colorbar(label="Exact $u_e(x,y)$")
plt.title("Exact Solution")

ax3=plt.subplot(1, 3, 3)
plt.contourf(X, Y, abs_error, levels=100, cmap="hot")
plt.colorbar(label="Absolute Error $|u_p - u_{e}|$")
plt.title("Error Map")

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

plt.tight_layout()
fig2.savefig("jack1.png")

fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Y,X, u_pred, cmap="viridis", edgecolor='none')
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Y,X, u_true, cmap="viridis", edgecolor='none')
ax2.set_title("Exact Solution")
ax1.set_title("Predicted Solution ")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("u_e")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u_p")


# Show the plots
plt.tight_layout()
plt.show()
fig.savefig("jack2.png")
