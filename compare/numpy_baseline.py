"""NumPy baseline for dart_ml_lab: the same experiment as bin/train.dart.

Requirement: numpy>=1.26  (Python 3.9+). No other dependencies.

    python compare/numpy_baseline.py

Everything that defines the experiment matches the Dart version: the spiral
formula and noise, the 80/20 split, the [2, 64, 64, 3] architecture, He-uniform
init, plain mini-batch SGD (batch 32, lr 0.2, 400 epochs, reshuffled each
epoch), the backprop formulas, and the inference and matmul benchmarks.

The seeds are the same integers, but NumPy's generator is not Dart's, so the
two programs draw different random numbers. Losses and accuracies will be
close, not identical. Each program is deterministic on its own.
"""

import sys
import time

import numpy as np

# Data.
POINTS_PER_CLASS = 300
CLASS_COUNT = 3
NOISE = 0.2
DATASET_SEED = 7
TRAIN_FRACTION = 0.8

# Model.
LAYER_SIZES = [2, 64, 64, 3]
MODEL_SEED = 42

# Training.
BATCH_SIZE = 32
LEARNING_RATE = 0.2
EPOCHS = 400
SHUFFLE_SEED = 1
LOG_EVERY = 50

# Inference benchmark.
WARMUP_CALLS = 1000
SINGLE_CALLS = 100_000
BATCH_ROWS = 10_000
BENCH_SEED = 99

# Matmul benchmark.
MATMUL_SIZES = [32, 64, 128, 256, 512]
MATMUL_SEED = 123
MIN_MEASURE_S = 0.2
WARMUP_S = 0.1


def make_spiral(points_per_class, classes, noise, rng):
    x = np.zeros((points_per_class * classes, 2))
    y = np.zeros(points_per_class * classes, dtype=np.int64)
    r = np.arange(points_per_class) / (points_per_class - 1)
    for c in range(classes):
        t = c * 4 + r * 4 + rng.standard_normal(points_per_class) * noise
        rows = slice(c * points_per_class, (c + 1) * points_per_class)
        x[rows, 0] = r * np.sin(t)
        x[rows, 1] = r * np.cos(t)
        y[rows] = c
    return x, y


def init_model(sizes, rng):
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        bound = np.sqrt(6 / fan_in)
        weights.append(rng.uniform(-bound, bound, size=(fan_in, fan_out)))
        biases.append(np.zeros((1, fan_out)))
    return weights, biases


def forward(weights, biases, x):
    """Returns every activation: [x, a1, ..., probs]."""
    activations = [x]
    a = x
    last = len(weights) - 1
    for l, (w, b) in enumerate(zip(weights, biases)):
        a = a @ w + b
        if l < last:
            np.maximum(a, 0, out=a)
        else:
            a -= a.max(axis=1, keepdims=True)
            np.exp(a, out=a)
            a /= a.sum(axis=1, keepdims=True)
        activations.append(a)
    return activations


def predict_proba(weights, biases, x):
    return forward(weights, biases, x)[-1]


def train_batch(weights, biases, x, labels, lr):
    batch = x.shape[0]
    activations = forward(weights, biases, x)
    probs = activations[-1]
    rows = np.arange(batch)
    loss = -np.log(np.maximum(probs[rows, labels], 1e-12)).mean()

    delta = probs.copy()
    delta[rows, labels] -= 1
    delta /= batch
    grads = []
    for l in range(len(weights) - 1, -1, -1):
        a_prev = activations[l]
        d_w = a_prev.T @ delta
        d_b = delta.sum(axis=0, keepdims=True)
        if l > 0:
            # Uses weights[l] before it is updated below.
            delta = (delta @ weights[l].T) * (a_prev > 0)
        grads.append((l, d_w, d_b))
    for l, d_w, d_b in grads:
        weights[l] -= lr * d_w
        biases[l] -= lr * d_b
    return loss


def accuracy(weights, biases, x, y):
    return float((predict_proba(weights, biases, x).argmax(axis=1) == y).mean())


def train_and_benchmark():
    data_rng = np.random.default_rng(DATASET_SEED)
    x, y = make_spiral(POINTS_PER_CLASS, CLASS_COUNT, NOISE, data_rng)
    order = data_rng.permutation(len(y))
    train_count = round(len(y) * TRAIN_FRACTION)
    train_idx, test_idx = order[:train_count], order[train_count:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    weights, biases = init_model(LAYER_SIZES, np.random.default_rng(MODEL_SEED))
    parameters = sum(w.size + b.size for w, b in zip(weights, biases))
    print(f"numpy baseline (Python {sys.version.split()[0]}, NumPy {np.__version__})")
    print(f"data:  {len(y)} points, {train_count} train / {len(test_idx)} test, "
          f"{CLASS_COUNT} classes, noise {NOISE}")
    print(f"model: {LAYER_SIZES}, {parameters} parameters")
    print(f"sgd:   batch {BATCH_SIZE}, lr {LEARNING_RATE}, {EPOCHS} epochs\n")

    shuffle_rng = np.random.default_rng(SHUFFLE_SEED)
    final_loss = 0.0
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        batch_order = shuffle_rng.permutation(train_count)
        loss_sum = 0.0
        for begin in range(0, train_count, BATCH_SIZE):
            idx = batch_order[begin:begin + BATCH_SIZE]
            loss = train_batch(weights, biases, x_train[idx], y_train[idx],
                               LEARNING_RATE)
            loss_sum += loss * len(idx)
        final_loss = loss_sum / train_count
        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(f"epoch {epoch:3d}  loss {final_loss:.6f}  "
                  f"train {accuracy(weights, biases, x_train, y_train):.2%}  "
                  f"test {accuracy(weights, biases, x_test, y_test):.2%}")
    train_ms = (time.perf_counter() - start) * 1000
    train_acc = accuracy(weights, biases, x_train, y_train)
    test_acc = accuracy(weights, biases, x_test, y_test)

    bench_rng = np.random.default_rng(BENCH_SEED)
    singles = [bench_rng.uniform(-1, 1, size=(1, LAYER_SIZES[0]))
               for _ in range(256)]
    sink = 0.0
    for i in range(WARMUP_CALLS):
        sink += predict_proba(weights, biases, singles[i & 255])[0, 0]
    start = time.perf_counter()
    for i in range(SINGLE_CALLS):
        sink += predict_proba(weights, biases, singles[i & 255])[0, 0]
    single_us = (time.perf_counter() - start) * 1e6 / SINGLE_CALLS

    big_batch = bench_rng.uniform(-1, 1, size=(BATCH_ROWS, LAYER_SIZES[0]))
    start = time.perf_counter()
    batch_probs = predict_proba(weights, biases, big_batch)
    batch_s = time.perf_counter() - start
    sink += float(batch_probs[:, 0].sum())

    print("\n=== summary ===")
    print(f"final loss:            {final_loss:.6f}")
    print(f"train accuracy:        {train_acc:.2%}")
    print(f"test accuracy:         {test_acc:.2%}")
    print(f"training time:         {train_ms:.1f} ms "
          f"({train_ms / EPOCHS:.3f} ms/epoch)")
    print(f"single prediction:     {single_us:.3f} us mean "
          f"over {SINGLE_CALLS} calls")
    print(f"batch of {BATCH_ROWS}:       {batch_s * 1000:.3f} ms total, "
          f"{batch_s * 1e6 / BATCH_ROWS:.3f} us/row")
    print(f"benchmark sink:        {sink:.6f}")
    print(f"parameters:            {parameters}")


def run_for(a, b, seconds):
    sink = 0.0
    reps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        sink += (a @ b)[0, 0]
        reps += 1
    return reps, time.perf_counter() - start, sink


def matmul_benchmark():
    print("\nmatmul benchmark (NumPy / BLAS), square n x n, float64")
    print(f"{'n':>5}  {'ms':>11}  {'GFLOPS':>10}")
    rng = np.random.default_rng(MATMUL_SEED)
    sink = 0.0
    for n in MATMUL_SIZES:
        a = rng.uniform(-1, 1, size=(n, n))
        b = rng.uniform(-1, 1, size=(n, n))
        run_for(a, b, WARMUP_S)
        calib_reps, calib_s, _ = run_for(a, b, MIN_MEASURE_S)
        reps = max(1, int(np.ceil(calib_reps * MIN_MEASURE_S / calib_s)))
        start = time.perf_counter()
        for _ in range(reps):
            sink += (a @ b)[0, 0]
        ms = (time.perf_counter() - start) * 1000 / reps
        print(f"{n:>5}  {ms:>11.4f}  {2.0 * n ** 3 / (ms / 1000) / 1e9:>10.3f}")
    print(f"sink: {sink:.6f}")


if __name__ == "__main__":
    train_and_benchmark()
    matmul_benchmark()
