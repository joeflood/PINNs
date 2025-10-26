import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the neural network architecture for the PINN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        # x: tensor of shape (N, 2) representing (x, y) coordinates
        a = x
        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        return self.layers[-1](a)

def f_func(x, y):
    # Define the source term for the Poisson equation:
    # f(x,y) = -2*pi^2*sin(pi*x)*sin(pi*y)
    pi = torch.pi
    return -2*(pi**2)*torch.sin(pi*x)*torch.sin(pi*y)

def u_exact_func(x, y):
    # Exact solution: u(x,y) = sin(pi*x)*sin(pi*y)
    pi = torch.pi
    return torch.sin(pi*x)*torch.sin(pi*y)

def pde_residual(model, x, y, f_func):
    """
    Compute the PDE residual for the Poisson equation:
    u_xx + u_yy - f(x,y) = 0.
    """
    # Ensure x and y require gradients
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    # Concatenate x and y into input tensor
    inputs = torch.cat([x, y], dim=1)
    u = model(inputs)
    
    # First derivatives: u_x and u_y
    grad_u = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    
    # Second derivatives: u_xx and u_yy
    u_xx = torch.autograd.grad(u_x, inputs, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, inputs, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0][:, 1:2]
    
    # Compute the residual: u_xx + u_yy - f(x, y)
    f_val = f_func(x, y)
    residual = u_xx + u_yy - f_val
    return residual

def boundary_loss(model, xb, yb, u_exact):
    """
    Compute the boundary loss by comparing the network's prediction to the exact solution.
    """
    inputs_b = torch.cat([xb, yb], dim=1)
    u_pred = model(inputs_b)
    loss_b = torch.mean((u_pred - u_exact)**2)
    return loss_b

def generate_training_data(N_f=3000, N_b=200):
    # Collocation points inside the domain [0,1] x [0,1]
    x_f = torch.rand(N_f, 1)
    y_f = torch.rand(N_f, 1)
    
    nb_side = N_b // 4
    
    # Left edge: x=0
    xb_left = torch.zeros(nb_side, 1)
    yb_left = torch.rand(nb_side, 1)
    
    # Right edge: x=1
    xb_right = torch.ones(nb_side, 1)
    yb_right = torch.rand(nb_side, 1)
    
    # Bottom edge: y=0
    xb_bottom = torch.rand(nb_side, 1)
    yb_bottom = torch.zeros(nb_side, 1)
    
    # Top edge: y=1
    xb_top = torch.rand(nb_side, 1)
    yb_top = torch.ones(nb_side, 1)
    
    xb = torch.cat([xb_left, xb_right, xb_bottom, xb_top], dim=0)
    yb = torch.cat([yb_left, yb_right, yb_bottom, yb_top], dim=0)
    
    # Compute exact boundary values using the exact solution
    u_b = u_exact_func(xb, yb)
    
    return x_f, y_f, xb, yb, u_b

def train(model, optimizer, epochs=5000):
    x_f, y_f, xb, yb, u_b = generate_training_data()
    
    loss_history = []             # 1) initialize history list
    
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        
        # PDE residual loss
        res = pde_residual(model, x_f, y_f, f_func)
        loss_f = torch.mean(res**2)
        
        # Boundary loss
        loss_b = boundary_loss(model, xb, yb, u_b)
        
        # Total loss
        loss = loss_f + loss_b
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())   # 2) record loss
        
        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{epochs} — Total Loss: {loss.item():.5e}')
    
    print("Training completed.")
    
    # 3) after training, plot loss vs epoch
    plt.figure(figsize=(6,4))
    plt.plot(loss_history, '-', linewidth=1)
    plt.yscale('log')              # often useful
    plt.xlabel('Epoch')
    plt.ylabel('Total Training Loss')
    plt.title('PINN Loss History')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define network architecture: 2 inputs (x, y), several hidden layers, 1 output (u)
    layers = [2, 50, 50, 50, 50, 50, 1]
    model = PINN(layers)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the PINN
    train(model, optimizer, epochs=5000)

# After training is completed
torch.save(model.state_dict(), 'pinn_poisson.pth')
print("Model saved to pinn_poisson.pth")
