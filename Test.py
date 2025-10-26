import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        a = x
        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        return self.layers[-1](a)

def u_exact_func(x, y):
    # Exact solution: u(x,y) = sin(pi*x)*sin(pi*y)
    pi = torch.pi
    return torch.sin(pi*x)*torch.sin(pi*y)

# Define the same network architecture as in training
layers = [2, 50, 50, 50, 50, 50, 1]
model = PINN(layers)

# Load the saved model parameters (ensure the file exists from your training run)
model.load_state_dict(torch.load('pinn_poisson.pth', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Generate a test grid over the domain [0,1] x [0,1]
nx, ny = 100, 100  # number of grid points in x and y directions
x = torch.linspace(0, 1, nx).reshape(-1, 1)
y = torch.linspace(0, 1, ny).reshape(-1, 1)
X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')

# Flatten the grid for batch evaluation (each row is a coordinate pair)
x_test = X.reshape(-1, 1)
y_test = Y.reshape(-1, 1)
inputs = torch.cat([x_test, y_test], dim=1)

# Evaluate the network's prediction at test points
with torch.no_grad():
    u_pred = model(inputs)

# Compute the exact solution at the test points
u_exact = u_exact_func(x_test, y_test)

# Calculate the relative L2 error
error = torch.norm(u_exact - u_pred, 2) / torch.norm(u_exact, 2)
print("L2 Relative Error: {:.2e}".format(error.item()))

# Reshape the predictions and exact solution for contour plotting
U_pred = u_pred.reshape(nx, ny).numpy()
U_exact = u_exact.reshape(nx, ny).numpy()
U_error = np.abs(U_pred - U_exact)

# Plot the predicted solution and the exact solution
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
contour1 = plt.contourf(X.numpy(), Y.numpy(), U_pred, 50, cmap='viridis')
plt.colorbar(contour1)
plt.title('PINN Prediction')

plt.subplot(1, 3, 2)
contour2 = plt.contourf(X.numpy(), Y.numpy(), U_exact, 50, cmap='viridis')
plt.colorbar(contour2)
plt.title('Exact Solution')

plt.subplot(1,3,3)
contour3 = plt.contourf(X.numpy(), Y.numpy(), U_error, 50, cmap='viridis')
plt.colorbar(contour3)
plt.title('Error')

plt.tight_layout()
plt.show()
