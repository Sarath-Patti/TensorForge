"""Training engine and training loop orchestrator for TensorForge."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from tensorforge.autograd.engine import no_grad
from tensorforge.data.dataloader import DataLoader
from tensorforge.nn.losses import CrossEntropyLoss
from tensorforge.nn.module import Module
from tensorforge.optim.optimizer import Optimizer
from tensorforge.tensor.tensor import Tensor
from tensorforge.training.metrics import accuracy


class Trainer:
    """Orchestrates model training, optimization, evaluation, and history logging.

    Args:
        model: The neural network Module to train.
        optimizer: Optimizer instance configured for model parameters.
        loss_fn: Loss function Module (e.g. MSELoss, CrossEntropyLoss).
        metrics: Optional dictionary mapping metric names to evaluation functions (f(pred, target) -> float).
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Module,
        metrics: Optional[Dict[str, Callable[[Tensor, Tensor], float]]] = None,
    ) -> None:
        self.model: Module = model
        self.optimizer: Optimizer = optimizer
        self.loss_fn: Module = loss_fn

        if metrics is None:
            if isinstance(loss_fn, CrossEntropyLoss):
                self.metrics: Dict[str, Callable[[Tensor, Tensor], float]] = {"acc": accuracy}
            else:
                self.metrics = {}
        else:
            self.metrics = metrics

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model for a specified number of epochs.

        Args:
            train_loader: DataLoader providing training batches.
            epochs: Number of complete passes over the dataset.
            val_loader: Optional DataLoader providing validation batches.
            verbose: If True, prints per-epoch training and validation metrics.

        Returns:
            Dictionary containing history of recorded training and validation metrics.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
        }
        for metric_name in self.metrics:
            history[f"train_{metric_name}"] = []

        if val_loader is not None:
            history["val_loss"] = []
            for metric_name in self.metrics:
                history[f"val_{metric_name}"] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            total_samples = 0
            running_metrics = {m: 0.0 for m in self.metrics}

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    x_batch, y_batch = batch[0], batch[1]
                else:
                    x_batch, y_batch = batch, batch

                batch_size = x_batch.shape[0] if x_batch.ndim > 0 else 1

                # Forward pass & loss
                self.optimizer.zero_grad()
                predictions = self.model(x_batch)
                loss = self.loss_fn(predictions, y_batch)

                # Backpropagation & optimization step
                loss.backward()
                self.optimizer.step()

                loss_val = float(loss.item())
                running_loss += loss_val * batch_size
                total_samples += batch_size

                # Track metrics
                for m_name, m_fn in self.metrics.items():
                    m_val = m_fn(predictions, y_batch)
                    running_metrics[m_name] += m_val * batch_size

            epoch_train_loss = running_loss / max(1, total_samples)
            history["train_loss"].append(epoch_train_loss)

            for m_name in self.metrics:
                m_avg = running_metrics[m_name] / max(1, total_samples)
                history[f"train_{m_name}"].append(m_avg)

            # Validation pass
            log_str = f"Epoch [{epoch:3d}/{epochs:3d}] - Loss: {epoch_train_loss:.4f}"
            for m_name in self.metrics:
                log_str += f" - {m_name}: {history[f'train_{m_name}'][-1]:.4f}"

            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                history["val_loss"].append(val_results["loss"])
                log_str += f" | Val Loss: {val_results['loss']:.4f}"
                for m_name in self.metrics:
                    val_m = val_results[m_name]
                    history[f"val_{m_name}"].append(val_m)
                    log_str += f" - Val {m_name}: {val_m:.4f}"

            if verbose:
                print(log_str)

        return history

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset in evaluation mode.

        Args:
            data_loader: DataLoader providing validation or test batches.

        Returns:
            Dictionary containing computed loss and metrics.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        running_metrics = {m: 0.0 for m in self.metrics}

        with no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    x_batch, y_batch = batch[0], batch[1]
                else:
                    x_batch, y_batch = batch, batch

                batch_size = x_batch.shape[0] if x_batch.ndim > 0 else 1

                predictions = self.model(x_batch)
                loss = self.loss_fn(predictions, y_batch)

                loss_val = float(loss.item())
                running_loss += loss_val * batch_size
                total_samples += batch_size

                for m_name, m_fn in self.metrics.items():
                    m_val = m_fn(predictions, y_batch)
                    running_metrics[m_name] += m_val * batch_size

        results: Dict[str, float] = {
            "loss": running_loss / max(1, total_samples)
        }
        for m_name in self.metrics:
            results[m_name] = running_metrics[m_name] / max(1, total_samples)

        return results
