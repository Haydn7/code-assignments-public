import unittest
import torch
from typing import Final
from provided.network import SimpleNeuralNetwork


class TestBackPropagation(unittest.TestCase):
    """"Tests the back propagation values calculated manually against those using torch autograd."""

    def setUp(self):
        self.TOL: Final[float] = 1.e-6

        # Construct the neural network and random sample data
        input_dim: Final[int] = 16
        hidden_sizes: Final[list[int]] = [64, 32]  # example hidden layer sizes
        output_size: Final[int] = 1
        batch_size: Final[int] = 8
        learning_rate: Final[float] = 1.e-4

        self.network = SimpleNeuralNetwork(input_dim, hidden_sizes, output_size)
        X = torch.randn(batch_size, input_dim)
        Y_target = torch.randn(batch_size, output_size)  # example target values for RMS loss

        # Back propagate first using the custom method then using auto grad
        Y = self.network.forward(X)
        self.network.backward(Y, learning_rate)
        loss = 0.5 * torch.sqrt(torch.mean((Y_target - Y) ** 2))
        loss.backward()

    def _test_close_gradients(self, name):
        actual = getattr(self.network, f"d{name}")
        target = getattr(self.network, name).grad
        self.assertTrue(torch.allclose(actual, target, atol=self.TOL), f"{name} gradients")

    def test_X_gradients(self): self._test_close_gradients("X")
    def test_b1_gradients(self): self._test_close_gradients("b1")
    def test_b2_gradients(self): self._test_close_gradients("b2")
    def test_b3_gradients(self): self._test_close_gradients("b3")
    def test_W1_gradients(self): self._test_close_gradients("W1")
    def test_W2_gradients(self): self._test_close_gradients("W2")
    def test_W3_gradients(self): self._test_close_gradients("W3")
    def test_Z1_gradients(self): self._test_close_gradients("Z1")
    def test_Z2_gradients(self): self._test_close_gradients("Z2")
    def test_Z3_gradients(self): self._test_close_gradients("Z3")


if __name__ == '__main__':
    unittest.main()
