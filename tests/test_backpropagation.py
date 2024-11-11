import unittest
import torch
from typing import Final
from provided.network import SimpleNeuralNetwork
from numpy.ma.testutils import assert_equal


class NetworkModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NetworkModelSingleton, cls).__new__(cls)
            cls._instance.data = cls._load_data()
        return cls._instance

    @staticmethod
    def _load_data():
        # Construct the neural network and random sample data
        input_dim: Final[int] = 16
        hidden_sizes: Final[list[int]] = [64, 32]  # example hidden layer sizes
        output_size: Final[int] = 1
        batch_size: Final[int] = 8

        network = SimpleNeuralNetwork(input_dim, hidden_sizes, output_size)
        X = torch.randn(batch_size, input_dim, requires_grad=True)
        Y_target = torch.randn(batch_size, output_size, requires_grad=True)

        # Back propagate using the custom method and auto grad
        params_requiring_gradients = "b1 b2 b3 W1 W2 W3".split()
        for p in params_requiring_gradients:
            param = getattr(network, p).detach()
            param.requires_grad_(True)
            if param.grad is not None:
                param.grad.zero_()
            setattr(network, p, param)

        Y = network.forward(X)
        loss = 0.5 * torch.sum((Y_target - Y) ** 2)
        loss.backward()

        gradients = { p: getattr(network, p).grad for p in params_requiring_gradients }
        gradients["X"] = X.grad
        network.calc_gradients(Y_target)
        return network, gradients

network_instance = NetworkModelSingleton()


class TestBackPropagation(unittest.TestCase):
    """"Tests the back propagation values calculated manually against those using torch autograd."""

    def setUp(self):
        self.network, self.gradients = network_instance.data
        self.TOL: Final[float] = 1.e-6

    def _test_close_gradients(self, name):
        actual = getattr(self.network, f"d{name}").detach()
        target = self.gradients[name]
        assert_equal(actual.shape, target.shape, f"{name} shape")
        self.assertTrue(torch.allclose(actual, target, atol=self.TOL), f"{name} gradients")

    def test_X_gradients(self): self._test_close_gradients("X")
    def test_b1_gradients(self): self._test_close_gradients("b1")
    def test_b2_gradients(self): self._test_close_gradients("b2")
    def test_b3_gradients(self): self._test_close_gradients("b3")
    def test_W1_gradients(self): self._test_close_gradients("W1")
    def test_W2_gradients(self): self._test_close_gradients("W2")
    def test_W3_gradients(self): self._test_close_gradients("W3")


if __name__ == '__main__':
    unittest.main()
