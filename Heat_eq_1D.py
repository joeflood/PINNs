import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture for the time-dependent problem
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        # x: tensor of shape (N, 2) where columns correspond to (x, t)
        a = x
        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        return self.layers[-1](a)

def f_func(x, t):
    # Define the source term:
    # f(x,t) = (pi^2 - 1)*exp(-t)*sin(pi*x)
    pi = torch.pi
    return (pi**2 - 1) * torch.exp(-t) * torch.sin(pi*x)

def u_exact_func(x, t):
    # Exact solution: u(x,t) = exp(-t)*sin(pi*x)
    pi = torch.pi
    return torch.exp(-t) * torch.sin(pi*x)

def pde_residual(model, x, t):
    # Enable gradients for x and t
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # Concatenate x and t to create the input tensor
    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)
    
    # Compute time derivative u_t
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    # Compute spatial derivative u_x and then second derivative u_xx
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    
    # Calculate the PDE residual: u_t - u_xx - f(x,t)
    f_val = f_func(x, t)
    residual = u_t - u_xx - f_val
    return residual

def boundary_loss(model, xb, tb, u_b):
    # Evaluate model on boundary points and compute loss against exact boundary values
    inputs = torch.cat([xb, tb], dim=1)
    u_pred = model(inputs)
    loss_b = torch.mean((u_pred - u_b)**2)
    return loss_b

def initial_loss(model, xi, ti, u_i):
    # Evaluate model at initial time t=0 and compare with the initial condition u(x,0)
    inputs = torch.cat([xi, ti], dim=1)
    u_pred = model(inputs)
    loss_i = torch.mean((u_pred - u_i)**2)
    return loss_i

def generate_training_data(N_f=10000, N_b=200, N_i=200):
    # Collocation points inside the spatio-temporal domain: x in (0,1), t in (0,1)
    x_f = torch.rand(N_f, 1)
    t_f = torch.rand(N_f, 1)
    
    # Boundary points: spatial boundaries (x=0 and x=1) for random t in [0,1]
    nb_side = N_b // 2
    xb_left = torch.zeros(nb_side, 1)
    xb_right = torch.ones(nb_side, 1)
    tb_left = torch.rand(nb_side, 1)
    tb_right = torch.rand(nb_side, 1)
    xb = torch.cat([xb_left, xb_right], dim=0)
    tb = torch.cat([tb_left, tb_right], dim=0)
    # For our exact solution, u(0,t) = 0 and u(1,t) = 0 because sin(0)=0 and sin(pi)=0
    u_b = torch.zeros_like(xb)
    
    # Initial condition: t=0 for x in [0,1]
    xi = torch.rand(N_i, 1)
    ti = torch.zeros(N_i, 1)
    u_i = u_exact_func(xi, ti)  # u(x,0) = sin(pi*x)
    
    return x_f, t_f, xb, tb, u_b, xi, ti, u_i

def train(model, optimizer, epochs=5000):
    x_f, t_f, xb, tb, u_b, xi, ti, u_i = generate_training_data()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute the PDE residual loss over collocation points
        res = pde_residual(model, x_f, t_f)
        loss_f = torch.mean(res**2)
        
        # Compute the loss on the spatial boundaries
        loss_b = boundary_loss(model, xb, tb, u_b)
        
        # Compute the loss for the initial condition
        loss_i = initial_loss(model, xi, ti, u_i)
        
        # Total loss: sum of all components
        loss = loss_f + loss_b + loss_i
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Total Loss: {loss.item():.5e}, PDE Loss: {loss_f.item():.5e}, ' +
                  f'Boundary Loss: {loss_b.item():.5e}, Initial Loss: {loss_i.item():.5e}')
    print("Training completed.")

if __name__ == "__main__":
    # Define network architecture: 2 inputs (x,t), several hidden layers, 1 output (u)
    layers = [2, 50, 50, 50, 1]
    model = PINN(layers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, epochs=5000)

torch.save(model.state_dict(), 'pinn_poisson.pth')
print("Model saved to pinn_poisson.pth")