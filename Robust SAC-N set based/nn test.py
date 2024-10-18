import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
def generate_data(n_samples=1000, n_features=3):
    X = np.random.rand(n_samples, n_features)
    Y = np.random.rand(n_samples, n_features)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Conditional Mean-Covariance Neural Network
class ConditionalMeanCovarianceNN(nn.Module):
    def __init__(self, n_features, n_targets):
        super(ConditionalMeanCovarianceNN, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Output heads: one for mean and one for covariance
        self.mean_head = nn.Linear(64, n_targets)
        self.covariance_head = nn.Linear(64, n_targets * (n_targets + 1) // 2)  # Predict lower-triangular elements

        # Initialize a learnable scale factor
        self.scale_factor = nn.Parameter(torch.tensor(1.0))  # Learnable parameter

    def forward(self, x):
        shared = self.shared_layers(x)
        mean = self.mean_head(shared)

        # Covariance: predict elements of the lower-triangular Cholesky decomposition
        cov_flat = self.covariance_head(shared)  # Shape: (batch_size, n_targets * (n_targets + 1) // 2)

        # Build lower-triangular matrix
        L = torch.zeros(x.size(0), n_targets, n_targets).to(x.device)
        indices = torch.tril_indices(row=n_targets, col=n_targets, offset=0)
        L[:, indices[0], indices[1]] = cov_flat

        # Scale the covariance matrix using the learnable scale factor
        scaled_cov = self.scale_factor.view(1, -1, 1) * (L @ L.transpose(1, 2))  # Reconstruct covariance and scale

        return mean, L, scaled_cov  # Return scaled covariance

# Custom loss function
def custom_loss(Y_pred, Y_true, L_pred, scaled_cov_pred):
    # Mean loss: Mean Squared Error between predicted and true Y
    mean_loss = nn.MSELoss()(Y_pred, Y_true)

    # Covariance loss: difference between predicted and empirical covariance
    # Compute residuals
    residuals = Y_pred - Y_true
    # Empirical covariance from residuals
    empirical_cov = torch.bmm(residuals.unsqueeze(2),
                               residuals.unsqueeze(1))  # Shape: (batch_size, n_targets, n_targets)

    # Covariance loss: MSE between predicted and empirical covariance matrices
    cov_loss = nn.MSELoss()(scaled_cov_pred, empirical_cov)  # Use scaled covariance here

    return mean_loss + cov_loss

# Training function with coverage enforcement
def train_model(model, X_train, Y_train, n_epochs=100, lr=0.001, coverage_target=0.90):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in range(n_epochs):
        coverage_count = 0
        model.train()
        optimizer.zero_grad()

        # Forward pass on the training data
        mean_pred, L_pred, scaled_cov_pred = model(X_train)

        # Compute training loss
        loss = custom_loss(mean_pred, Y_train, L_pred, scaled_cov_pred)
        train_losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate coverage of training points (inside ellipsoid)
        for i in range(X_train.shape[0]):
            point = Y_train[i]
            mean = mean_pred[i]
            cov = scaled_cov_pred[i]

            # Check if the point is inside the ellipsoid
            if (point - mean) @ torch.linalg.inv(cov) @ (point - mean).T <= 1:
                coverage_count += 1

        # Calculate coverage percentage
        coverage_percentage = (coverage_count / len(Y_train)) * 100

        # If coverage is below target, apply penalty
        if coverage_percentage < coverage_target * 100:
            loss += (coverage_target * len(Y_train) - coverage_count) * 1e-3  # Adjust the penalty weight as needed

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item()}, Coverage: {coverage_percentage:.2f}%")

    return train_losses

# Visualization function for 2D subplots
def plot_points_and_ellipsoids(model, n_points=5, test_data=None):
    with torch.no_grad():
        # Generate random input points
        random_inputs = torch.rand(n_points, n_features)

        # Get mean and scaled covariance for random inputs
        means, _, scaled_covariances = model(random_inputs)

        # Create subplots
        fig, axs = plt.subplots(1, n_points, figsize=(5 * n_points, 5))

        # Count points inside ellipsoids
        points_inside_count = 0

        # Plot each point and its ellipsoid in separate subplots
        for i in range(n_points):
            mean = means[i].numpy()[:2]  # Take only the first two dimensions
            cov = scaled_covariances[i].numpy()[:2, :2]  # Take the first two rows and columns

            # Create a grid of points for the ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Create a 2D ellipsoid
            theta = np.linspace(0, 2 * np.pi, 100)
            ellipse = np.sqrt(eigenvalues)[:, None] * eigenvectors @ np.array([np.cos(theta), np.sin(theta)])

            # Plot mean point
            axs[i].scatter(mean[0], mean[1], color='r', s=1)  # Point in red
            # Plot ellipsoid
            axs[i].plot(mean[0] + ellipse[0], mean[1] + ellipse[1], alpha=0.5)

            # Plot original test points
            axs[i].scatter(test_data[i, 0], test_data[i, 1], color='b', s=1)  # Test points in blue

            # Check points inside ellipsoid
            for point in test_data:
                # Check if point is inside the ellipsoid
                point_shifted = point[:2] - mean
                if (point_shifted @ np.linalg.inv(cov) @ point_shifted.T) <= 1:
                    points_inside_count += 1

            axs[i].set_xlim(mean[0] - 1, mean[0] + 1)
            axs[i].set_ylim(mean[1] - 1, mean[1] + 1)
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
            axs[i].set_title(f'Point {i + 1}')
            axs[i].set_aspect('equal')

        plt.suptitle('Random Points and Corresponding Ellipsoids')
        plt.tight_layout()
        plt.show()

        # Calculate and print the percentage of points inside the ellipsoids
        percentage_inside = (points_inside_count / (n_points * len(test_data))) * 100
        print(f"Percentage of test points inside ellipsoids: {percentage_inside:.2f}%")

# Main script
if __name__ == "__main__":
    n_features = 3  # Number of input features
    n_targets = 3  # Number of output targets

    # Generate synthetic training data
    X_train, Y_train = generate_data(n_samples=1000, n_features=n_features)

    # Initialize and train the model
    model = ConditionalMeanCovarianceNN(n_features, n_targets)
    train_losses = train_model(model, X_train, Y_train, n_epochs=100, lr=0.001, coverage_target=0.90)

    # Plot training loss
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    # Display learned scaling factor
    print(f"Learned scale factor: {model.scale_factor.item()}")

    # Generate test data for plotting
    test_data, _ = generate_data(n_samples=100, n_features=n_features)  # 100 test points

    # Plot random points and their corresponding ellipsoids in 2D
    plot_points_and_ellipsoids(model, n_points=5, test_data=test_data)
