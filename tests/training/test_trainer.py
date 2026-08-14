"""Unit tests for Trainer loop and evaluation in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import float32, int64, tensor
from tensorforge.data import DataLoader, TensorDataset
from tensorforge.nn import CrossEntropyLoss, Linear, MSELoss, ReLU, Sequential
from tensorforge.optim import SGD, Adam
from tensorforge.training import Trainer


class TestTrainer(unittest.TestCase):
    """Tests for Trainer loop, parameter updating, loss reduction, and evaluation."""

    def test_linear_regression_training_fit(self):
        # Synthetic dataset: y = 2*x1 - 3*x2 + 1
        np.random.seed(42)
        X_np = np.random.uniform(-2, 2, (100, 2)).astype(np.float32)
        y_np = (2.0 * X_np[:, 0] - 3.0 * X_np[:, 1] + 1.0).reshape(-1, 1).astype(np.float32)

        dataset = TensorDataset(tensor(X_np), tensor(y_np))
        loader = DataLoader(dataset, batch_size=16, shuffle=True, seed=42)

        model = Linear(2, 1, bias=True)
        initial_weight = model.weight.numpy().copy()

        opt = SGD(model.parameters(), lr=0.05)
        loss_fn = MSELoss()

        trainer = Trainer(model, opt, loss_fn)
        history = trainer.fit(loader, epochs=15, verbose=False)

        # Loss should decrease significantly
        self.assertIn("train_loss", history)
        self.assertTrue(history["train_loss"][-1] < history["train_loss"][0])
        self.assertTrue(history["train_loss"][-1] < 0.1)

        # Weights should have changed
        self.assertFalse(np.allclose(model.weight.numpy(), initial_weight))

    def test_classification_training_with_validation(self):
        # 2-class synthetic dataset
        np.random.seed(123)
        X_train = np.random.randn(80, 4).astype(np.float32)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)

        X_val = np.random.randn(20, 4).astype(np.float32)
        y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)

        train_loader = DataLoader(TensorDataset(tensor(X_train), tensor(y_train)), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(tensor(X_val), tensor(y_val)), batch_size=20, shuffle=False)

        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2),
        )
        opt = Adam(model.parameters(), lr=0.05)
        loss_fn = CrossEntropyLoss()

        trainer = Trainer(model, opt, loss_fn)
        history = trainer.fit(train_loader, epochs=10, val_loader=val_loader, verbose=False)

        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("train_acc", history)
        self.assertIn("val_acc", history)
        self.assertEqual(len(history["train_loss"]), 10)
        self.assertEqual(len(history["val_loss"]), 10)

    def test_evaluation_mode_no_grad(self):
        model = Linear(2, 2)
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(model, opt, loss_fn)

        x = tensor([[1.0, 2.0]], dtype=float32)
        y = tensor([0], dtype=int64)
        loader = DataLoader(TensorDataset(x, y), batch_size=1)

        eval_res = trainer.evaluate(loader)
        self.assertIn("loss", eval_res)
        self.assertIn("acc", eval_res)

        # Confirm evaluation does not set gradients on model parameters
        for p in model.parameters():
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
