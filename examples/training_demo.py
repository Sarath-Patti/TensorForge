"""End-to-end Neural Network Training Demonstration for TensorForge v0.4.

Demonstrates:
1. Generating a deterministic synthetic 2D binary classification dataset.
2. Wrapping data with TensorDataset and mini-batch DataLoader.
3. Defining a multi-layer perceptron with Sequential, Linear, and ReLU.
4. Setting up Adam optimizer and CrossEntropyLoss.
5. Training the model across epochs using the Trainer engine.
6. Evaluating on validation data and displaying training metrics history.
"""

import numpy as np

import tensorforge as tf
from tensorforge.data import DataLoader, TensorDataset
from tensorforge.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from tensorforge.optim import Adam
from tensorforge.training import Trainer


def generate_synthetic_data(n_samples: int = 300, seed: int = 42):
    """Generate two concentric noisy circles for classification."""
    np.random.seed(seed)
    n_per_class = n_samples // 2

    # Inner circle (class 0)
    theta0 = np.random.uniform(0, 2 * np.pi, n_per_class)
    r0 = np.random.normal(1.0, 0.2, n_per_class)
    x0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    y0 = np.zeros(n_per_class, dtype=np.int64)

    # Outer circle (class 1)
    theta1 = np.random.uniform(0, 2 * np.pi, n_per_class)
    r1 = np.random.normal(2.5, 0.2, n_per_class)
    x1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    y1 = np.ones(n_per_class, dtype=np.int64)

    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate([y0, y1])

    # Shuffle
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]


def main() -> None:
    print("=== TensorForge v0.4 - Optimizers & Training Engine Demo ===")
    print(f"TensorForge Version: {tf.__version__}\n")

    # 1. Generate Synthetic Data
    print("1. Generating Synthetic Classification Dataset (Concentric Circles)")
    X, y = generate_synthetic_data(n_samples=400, seed=42)

    # Split into 80% train / 20% validation
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Feature dimensions: {X_train.shape[1]}, Classes: 2\n")

    # 2. Create Datasets and DataLoaders
    print("2. Constructing TensorDataset and DataLoaders")
    train_dataset = TensorDataset(tf.tensor(X_train), tf.tensor(y_train))
    val_dataset = TensorDataset(tf.tensor(X_val), tf.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"Train batches per epoch: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # 3. Define Neural Network Model
    print("3. Building Neural Network Model Architecture")
    model = Sequential(
        Linear(in_features=2, out_features=16, bias=True),
        ReLU(),
        Linear(in_features=16, out_features=8, bias=True),
        ReLU(),
        Linear(in_features=8, out_features=2, bias=True),
    )
    print("Model Summary:")
    print(model)
    print()

    # 4. Configure Optimizer and Loss Function
    print("4. Configuring Loss Function and Adam Optimizer")
    loss_fn = CrossEntropyLoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    print(f"Loss Function: {loss_fn}")
    print(f"Optimizer: {optimizer}\n")

    # 5. Train Model with Trainer
    print("5. Training Model with TensorForge Trainer...")
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
    history = trainer.fit(
        train_loader=train_loader,
        epochs=15,
        val_loader=val_loader,
        verbose=True,
    )
    print()

    # 6. Final Evaluation
    print("6. Final Evaluation on Validation Dataset:")
    val_results = trainer.evaluate(val_loader)
    print(f"Final Validation Loss:     {val_results['loss']:.4f}")
    print(f"Final Validation Accuracy: {val_results['acc'] * 100:.2f}%\n")

    print("Training successfully completed using TensorForge v0.4!")


if __name__ == "__main__":
    main()
