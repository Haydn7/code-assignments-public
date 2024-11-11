import einops
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class SimpleNeuralNetwork:
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_size: int) -> None:
        """
        Initialize weights and biases for the neural network layers.
        
        input_size: The number of input features.
        hidden_sizes: A list of integers representing the size of each hidden layer.
        output_size: The number of output neurons.
        """

        if len(hidden_sizes) != 2:
            raise ValueError("Hidden sizes must be size 2")

        # Initialize weights and biases for the layers
        self.W1: torch.tensor = nn.init.xavier_normal_(
            torch.empty(hidden_sizes[0], input_dim)
        )
        self.W2: torch.tensor = nn.init.xavier_normal_(
            torch.empty(hidden_sizes[1], hidden_sizes[0])
        )
        self.W3: torch.tensor = nn.init.xavier_normal_(
            torch.empty(output_size, hidden_sizes[1])
        )

        self.b1: torch.tensor = torch.zeros((1, hidden_sizes[0]))
        self.b2: torch.tensor = torch.zeros((1, hidden_sizes[1]))
        self.b3: torch.tensor = torch.zeros((1, output_size))

        # These will be set during the forward pass
        self.X: torch.tensor = None
        self.Z1: torch.tensor = None
        self.A1: torch.tensor = None
        self.Z2: torch.tensor = None
        self.A2: torch.tensor = None
        self.Z3: torch.tensor = None
        self.output: torch.tensor = None

        # These will be set during the backward pass
        self.dX: torch.tensor = None
        self.dZ1: torch.tensor = None
        self.dZ2: torch.tensor = None
        self.dZ3: torch.tensor = None
        self.doutput: torch.tensor = None


    def forward(self, X) -> np.ndarray:
        """
        Perform a forward pass through the network.
        
        X: The input data as a NumPy array. Example: X.shape = (m, n)
        Returns the output of the network after the forward pass.
        """
        self.X = X
        self.Z1 = einops.einsum(self.W1, X, "hidden features, batches features -> batches hidden") + self.b1
        self.Z2 = einops.einsum(self.W2, self.Z1, "hidden features, batches features -> batches hidden") + self.b2
        self.output = einops.einsum(self.W3, self.Z2, "hidden features, batches features -> batches hidden") + self.b3

        return self.output
    
    def calculate_accuracy(self, dataloader: DataLoader) -> float:
        """
        Calculate the accuracy of the network on a given dataloader.
        For binary classification, uses 0.5 as the threshold.

        Args:
            dataloader: DataLoader containing test data

        Returns:
            float: Accuracy as a percentage (0-100)
        """
        correct = 0
        total = 0

        # Set network to evaluation mode (if it supports eval)
        if hasattr(self, 'eval'):
            self.eval()

        with torch.no_grad():  # No need to track gradients for evaluation
            for X, labels in tqdm(dataloader, desc='Calculating accuracy'):
                # Forward pass
                outputs = self.forward(X)

                # Convert outputs to predictions (threshold at 0.5 for binary classification)
                predictions = (outputs >= 0.5).float()

                # Compare predictions with labels
                correct += torch.sum(predictions == labels.view(-1, 1))
                total += labels.size(0)

        accuracy = (correct / total) * 100
        return accuracy.item()
    
    def train(self, train_dataloader: DataLoader, epochs: int=2, batch_size: int=32, learning_rate: float=0.01) -> List[float]:
        """
        Train the neural network.

        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for each training step.
            learning_rate (float): Learning rate for weight updates.

        Returns:
            List[float]: List of average loss values per epoch.
        """
        losses = []

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []

            # Use tqdm for progress bar
            with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
                for _, (X, labels) in enumerate(pbar):

                    if X.shape[0] != batch_size:
                        raise ValueError(f'Expected batch size {batch_size}, got {X.shape[0]}')

                    # Reshape to (batch_size, 1)
                    X = X.float()
                    Y = labels.float().reshape(-1, 1)  

                    # Forward pass
                    output = self.forward(X)

                    # Compute loss (Mean Squared Error)
                    loss = torch.mean((output - Y) ** 2)
                    epoch_losses.append(loss)

                    # Backward pass
                    self.backward(Y, learning_rate)

                    # Update progress bar with current loss
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Average loss for this epoch
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f'\nEpoch {epoch + 1} average loss: {avg_loss:.4f}')
    
    def backward(self, Y: torch.Tensor, learning_rate: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a backward pass to compute gradients and update weights.

        Args:
            Y: Target output (batch_size, output_size)

        Returns:
            Tuple of (dX, dA1, dA2, dZ3)
        """

        # Output layer gradient ()
        self.calc_gradients(Y)

        # Update weights and biases
        self.W3 -= learning_rate * self.dW3
        self.W2 -= learning_rate * self.dW2
        self.W1 -= learning_rate * self.dW1
        self.b3 -= learning_rate * self.db3
        self.b2 -= learning_rate * self.db2
        self.b1 -= learning_rate * self.db1

        return self.dX, self.dZ1, self.dZ2, self.dZ3


    def calc_gradients(self, Y: torch.Tensor) -> None:
        """
        Calculates the gradients.
        Separated out of backward to enable unit testing of gradients (as backward update the parameters)

        Args:
            Y: Target output (batch_size, output_size)

        Returns:
            None
        """

        # Assuming mean squared error loss
        self.dZ3 = self.output - Y  # (batch_size, output_size)

        # Just rearrange the einsum to back propagate the gradients
        #  dY has shape (batches, hidden), W shape (hidden, features), X shape (batches, features)
        self.dZ2 = einops.einsum(self.W3, self.dZ3, "hidden features, batches hidden -> batches features")
        self.dZ1 = einops.einsum(self.W2, self.dZ2, "hidden features, batches hidden -> batches features")
        self.dX =  einops.einsum(self.W1, self.dZ1, "hidden features, batches hidden -> batches features")

        self.db3 = einops.einsum(self.dZ3, "batches hidden -> hidden").unsqueeze(0) # Match shape of bias [1, channels]
        self.db2 = einops.einsum(self.dZ2, "batches hidden -> hidden").unsqueeze(0)
        self.db1 = einops.einsum(self.dZ1, "batches hidden -> hidden").unsqueeze(0)

        self.dW3 = einops.einsum(self.Z2, self.dZ3, "batches features, batches hidden -> hidden features")
        self.dW2 = einops.einsum(self.Z1, self.dZ2, "batches features, batches hidden -> hidden features")
        self.dW1 =  einops.einsum(self.X, self.dZ1, "batches features, batches hidden -> hidden features")