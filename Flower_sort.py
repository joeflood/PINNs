import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features: [Sepal Length, Sepal Width, Petal Length, Petal Width]
y = iris.target  # Labels: 0 (Setosa), 1 (Versicolor), 2 (Virginica)

# 🔹 Select only two classes: Setosa (0) & Versicolor (1)
mask = (y == 0) | (y == 1)  # Keep only labels 0 and 1
X = X[mask]
y = y[mask]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Make 2D for BCE
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define a simple binary classifier
class BinaryIrisClassifier(nn.Module):
    def __init__(self):
        super(BinaryIrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 4 input features -> 16 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # 16 neurons -> 1 output neuron (binary classification)
        self.sigmoid = nn.Sigmoid()  # Output probability (0 to 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply sigmoid to get probabilities
        return x

# Create model, loss function, and optimizer
model = BinaryIrisClassifier()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predictions = (test_outputs >= 0.5).float()  # Convert probabilities to 0 or 1
    accuracy = (predictions == y_test_tensor).sum().item() / y_test_tensor.size(0)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

# Get model predictions on the test set
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predictions = (test_outputs >= 0.5).float()  # Convert probabilities to 0 or 1

# Convert tensors to NumPy for plotting
X_test_np = X_test_tensor.numpy()
y_test_np = y_test_tensor.numpy()
predictions_np = predictions.numpy()

# Create a mesh grid for petal length and petal width
x_min, x_max = X_test_np[:, 2].min() - 0.5, X_test_np[:, 2].max() + 0.5
y_min, y_max = X_test_np[:, 3].min() - 0.5, X_test_np[:, 3].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Prepare grid data for model prediction (using only petal length & petal width)
grid_data = np.c_[np.zeros((xx.size, 2)), xx.ravel(), yy.ravel()]  # Set sepal features to 0
grid_tensor = torch.tensor(grid_data, dtype=torch.float32)

# Get model predictions for the entire grid
with torch.no_grad():
    grid_predictions = (model(grid_tensor) >= 0.5).float().numpy().reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, grid_predictions, levels=1, cmap="coolwarm", alpha=0.3)

# Plot test data points
plt.scatter(X_test_np[:, 2], X_test_np[:, 3], c=predictions_np.flatten(), cmap="coolwarm", edgecolors="k")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Decision Boundary for Iris Classification")
plt.colorbar(label="Predicted Class (0=Setosa, 1=Versicolor)")
plt.show()
