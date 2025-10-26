import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# For visualization
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        return self.net(inputs)

def pde_residual(model, x, t, nu):
    u = model(x, t)
    
    # Compute derivatives using autograd
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Burgers' equation residual
    residual = u_t + u * u_x - nu * u_xx
    return residual

# Define training data (collocation points, boundary conditions)
x_collocation = torch.rand(1000, 1, requires_grad=True) * 2 - 1  # Random points in [-1,1]
t_collocation = torch.rand(1000, 1, requires_grad=True) * 1      # Random points in [0,1]

# Initial condition u(x,0) = -sin(pi x)
x_init = torch.linspace(-1, 1, 100).view(-1, 1)
t_init = torch.zeros_like(x_init)
u_init = -torch.sin(np.pi * x_init)

# Boundary condition u(-1,t) = u(1,t) = 0
x_boundary = torch.cat([torch.ones(100, 1) * -1, torch.ones(100, 1) * 1])
t_boundary = torch.rand(200, 1)
u_boundary = torch.zeros_like(x_boundary)

# Define model, optimizer
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
nu = 0.01  # viscosity

# Training loop
for epoch in range(5000):
    optimizer.zero_grad()

    # Compute loss components
    residual_loss = torch.mean(pde_residual(model, x_collocation, t_collocation, nu) ** 2)
    ic_loss = torch.mean((model(x_init, t_init) - u_init) ** 2)
    bc_loss = torch.mean((model(x_boundary, t_boundary) - u_boundary) ** 2)

    # Total loss
    loss = residual_loss + ic_loss + bc_loss

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

x_test = torch.linspace(-1, 1, 100).view(-1, 1)
t_test = torch.linspace(0, 1, 100).view(-1, 1)
X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze())

u_pred = model(X.reshape(-1, 1), T.reshape(-1, 1)).detach().numpy()
u_pred = u_pred.reshape(100, 100)

plt.imshow(u_pred, extent=(-1, 1, 0, 1), origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('PINN Solution for Burgers’ Equation')
plt.show()
