# dart_ml_lab

A neural network written from scratch in pure Dart, trained on a toy dataset and
served over HTTP.

This is an experiment, not a product. The question is: **can Dart be used to
build and serve an AI model from scratch, and what are its real strengths and
weaknesses compared to Python/NumPy?** Everything numeric is plain Dart: no ML
or numeric packages, no FFI, no native bindings. The only dependencies are
`shelf` and `shelf_router` for the HTTP server. The goal is measurement, so the
results section reports what was actually run on one machine, including the
places where the numbers disagree with what I expected.

## Architecture

```
            TRAIN                                        SERVE
   ┌────────────────────┐                      ┌────────────────────┐
   │  bin/train.dart    │                      │  bin/server.dart   │   application
   │  (data, SGD loop,  │ ──▶  model.json ──▶  │  (shelf, isolates, │   layer: all I/O
   │   benchmark, file) │      the contract    │   validation)      │
   └─────────┬──────────┘                      └─────────┬──────────┘
             │                                           │
             ▼                                           ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  lib/mlp.dart      MLP: forward, backprop, SGD, toJson/fromJson │   model layer
   │  lib/dataset.dart  spiral generator, Box-Muller Gaussian        │   no file/HTTP code
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  lib/matrix.dart   Matrix over Float64List: matmul, softmax…    │   math layer
   └──────────────────────────────────────────────────────────────┘   pure, no I/O
```

Dependencies point downward only. `train.dart` and `server.dart` never import
each other; they communicate only through `model.json`. The server loads the
weights once, never trains and never mutates them, so each isolate is read-only
and stateless, which is what makes running N of them trivial.

`bin/bench_matmul.dart` and `bin/load_test.dart` are measurement tools.
`compare/numpy_baseline.py` is the same experiment in NumPy.

### Key decisions

| Decision | Reason |
|---|---|
| `Float64List`, row-major, one allocation per matrix | Raw contiguous 8-byte doubles. A `List<double>` can hold boxed values, which costs a pointer chase per element. Element `(r, c)` is `data[r * cols + c]`. |
| i-k-j loop order in `matmul` | The inner loop reads a row of `B` and writes a row of the output, both sequentially. i-j-k strides down a column of `B`. `matmulNaive` keeps the i-j-k version for the benchmark only. (The measured benefit on this machine is smaller than expected; see Findings.) |
| Batched matrix math, not scalar autograd | A layer is one `matmul` over the whole batch. A scalar autograd graph would allocate an object per multiply; here a training step is about ten tight loops over typed arrays. |
| Manual backprop | For softmax + cross-entropy, `delta = (probs − one_hot) / batch`. Then per layer, going backward: `dW = A_prevᵀ · delta`, `db = sumRows(delta)`, `delta_prev = (delta · Wᵀ) ⊙ [A_prev > 0]`. `computeGradients` returns all gradients without touching the weights and `trainBatch` applies them afterwards, so `delta_prev` is always computed from the weights the forward pass used. |
| He-uniform init, bound `sqrt(6 / fan_in)` | Keeps activation variance stable through ReLU layers (`Var(w) = 2 / fan_in`). Biases start at zero. |
| Stable softmax | Each row's max is subtracted before `exp`, so the largest exponent is 0 and nothing overflows. `[1000, 1001, 1002]` is a test case. Probabilities are clamped to `1e-12` before `log`. |
| Gradient check | Hand-written backprop is easy to get subtly wrong and still "train". The test compares every analytic weight and bias gradient with a central finite difference (`ε = 1e-5`) and requires relative error `< 1e-5`. Measured max error: `2.9e-9` over 54 parameters. |
| Fixed seeds everywhere | Dataset, split, init, epoch shuffling and benchmark inputs each use a seeded `Random`. Repeated runs produce identical logs and a byte-identical `model.json`, under both JIT and AOT. |

## Running it

```bash
dart pub get
dart test                         # 47 tests, including the gradient check
dart analyze

dart run bin/train.dart           # trains, benchmarks inference, writes model.json

dart run bin/server.dart --port 8080 --isolates 1
curl -s localhost:8080/health
curl -s -X POST localhost:8080/predict -d '{"inputs": [[0.5, 0.5], [-0.3, 0.8]]}'

dart run bin/bench_matmul.dart
dart run bin/load_test.dart --url http://127.0.0.1:8080/predict \
    --concurrency 50 --requests 5000 --batch 1

python3 -m venv .venv && .venv/bin/python -m pip install "numpy>=1.26"
.venv/bin/python compare/numpy_baseline.py
```

AOT builds:

```bash
mkdir -p build
dart compile exe bin/train.dart -o build/train
dart compile exe bin/server.dart -o build/server
dart compile exe bin/bench_matmul.dart -o build/bench_matmul
dart compile exe bin/load_test.dart -o build/load_test
./build/server --isolates 11
```

Run everything from the project root: `model.json` is read and written relative
to the working directory. The server listens on `127.0.0.1` only.

### API

`GET /health` → `{"status":"ok","parameters":4547,"isolate":0}`

`POST /predict` with `{"inputs": [[x, y], ...]}` →

```json
{"predictions":[{"class":1,"probabilities":[0.0000566,0.9999434,4.29e-14]}],
 "count":1,"inference_us":6}
```

`inference_us` covers only the model call. Invalid JSON, a missing `inputs`
field, an empty list, more than 10,000 rows, a row of the wrong length, and a
non-numeric or non-finite value each return `400` with `{"error": "..."}`.
Unexpected errors return a generic `500`; stack traces go to stderr only.

## Results

### Machine

| | |
|---|---|
| CPU | Apple M4 Max, 11 cores (5 performance + 6 efficiency) |
| RAM | 36 GB |
| OS | macOS 26.6.2 (arm64) |
| Dart | 3.13.3 stable |
| Python | 3.13.7 |
| NumPy | 2.5.3, linked against Apple Accelerate (multithreaded BLAS), in a local `.venv` |

All numbers are from 30 September 2026. The machine was not idle (load average
about 5 from other applications), so treat differences of a few percent as
noise. Where a range is given it is min–max over the runs stated.

### Training (spiral, 900 points, `[2, 64, 64, 3]`, 4,547 parameters)

Default hyperparameters, no tuning: batch 32, learning rate 0.2, 400 epochs.

| | Dart JIT | Dart AOT | NumPy |
|---|---|---|---|
| Final loss | 0.014706 | 0.014706 | 0.029363 |
| Train accuracy | 99.72% | 99.72% | 99.31% |
| **Test accuracy** | **98.89%** | **98.89%** | **100.00%** |
| Training time (3 runs) | 4,286–4,327 ms | 4,098–4,123 ms | 528–544 ms |
| Per epoch | 10.8 ms | 10.3 ms | 1.3 ms |
| Single prediction, mean of 100,000 | 4.79–4.90 µs | 4.73–4.78 µs | 8.41–8.85 µs |
| Batch of 10,000 rows | 44.0–48.4 ms (4.4–4.8 µs/row) | 47.3–47.8 ms (4.7–4.8 µs/row) | 3.03–3.12 ms (0.30–0.31 µs/row) |

JIT and AOT produce the same losses, accuracies and `model.json` (same SHA-1),
run after run. The NumPy run is also deterministic, but its random generator is
not Dart's, so the same seed integers give different data, a different split
and different initial weights. The accuracy difference between the two columns
(98.89% vs 100.00% on 180 test points, i.e. two points) reflects that, not a
difference in the algorithm.

### Matmul (`n × n`, Float64, single thread)

GFLOPS = `2n³ / time`. Two runs each; the second is shown, the first was within
about 8%.

| n | JIT naive ms | JIT naive GFLOPS | JIT i-k-j ms | JIT i-k-j GFLOPS | AOT naive ms | AOT naive GFLOPS | AOT i-k-j ms | AOT i-k-j GFLOPS |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.0390 | 1.68 | 0.0334 | 1.96 | 0.0304 | 2.15 | 0.0311 | 2.11 |
| 64 | 0.3024 | 1.73 | 0.2468 | 2.12 | 0.2310 | 2.27 | 0.2413 | 2.17 |
| 128 | 2.4957 | 1.68 | 2.1915 | 1.91 | 1.9957 | 2.10 | 2.0127 | 2.08 |
| 256 | 18.378 | 1.83 | 15.691 | 2.14 | 15.785 | 2.13 | 15.300 | 2.19 |
| 512 | 175.57 | 1.53 | 126.53 | 2.12 | 141.36 | 1.90 | 126.24 | 2.13 |

NumPy (`a @ b`, Accelerate, all cores), 3 runs:

| n | NumPy ms | NumPy GFLOPS | vs Dart AOT i-k-j |
|---|---|---|---|
| 32 | 0.0013 | 50–51 | ~24× |
| 64 | 0.0029 | 179–181 | ~83× |
| 128 | 0.0142–0.0145 | 290–294 | ~140× |
| 256 | 0.0938–0.0940 | 357–358 | ~163× |
| 512 | 0.75–0.78 | 343–357 | ~165× |

This is not a like-for-like comparison of languages: the Dart numbers are one
thread of scalar code, the NumPy numbers are a hand-tuned vendor BLAS using
every core and the CPU's vector units. It is the comparison that matters in
practice, though.

### Server (AOT, 50 concurrent connections, 3 runs each)

I used 50,000 requests for batch 1 and 20,000 for batch 100 rather than the
default 5,000, because 5,000 requests finish in about a quarter of a second.
Server request logs were redirected to a file. Zero errors in every run.

| Isolates | Batch | Throughput (req/s) | Rows/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| 1 | 1 | 20,917–21,337 | ~21,000 | 2.36–2.92 | 2.54–3.14 | 3.14–3.37 |
| 11 | 1 | 34,571–35,022 * | ~35,000 | 1.40–1.42 | 1.63–1.78 | 1.94–2.84 |
| 1 | 100 | 1,511–1,532 | ~152,000 | 33.1–39.5 | 40.1–47.5 | 43.4–50.8 |
| 11 | 100 | 7,921–8,190 | ~800,000 | 4.0–5.0 | 12.6–17.6 | 27.2–31.5 |

\* Limited by the load generator, not the server. `load_test.dart` is itself a
single isolate. Running two generators at once against the 11-isolate server
gave 26,933 + 26,271 = 53,204 req/s in total.

The server reports `inference_us` of about 6 µs for one row and about 400 µs for
100 rows.

### Startup and size

| | |
|---|---|
| `build/server` | 6.50 MB |
| `build/train` | 5.55 MB |
| `build/bench_matmul` | 5.45 MB |
| `build/load_test` | 6.08 MB |
| `model.json` | 89.9 KB for 4,547 parameters |
| Cold start, AOT, 1 isolate (process start → first 200 from `/health`, 20 runs) | median 24.5 ms, min 14.5 ms, max 41.0 ms |
| Cold start, AOT, 11 isolates (20 runs) | median 26.5 ms, min 14.7 ms, max 30.7 ms |
| Cold start, `dart run bin/server.dart` (JIT, 5 runs) | median 999 ms |
| Server RSS after load test | ~50 MB (1 isolate), ~55 MB (11 isolates) |

The AOT binaries are self-contained: they include the Dart runtime and need no
SDK on the target machine.

## Findings

### What the data supports

- **Correctness is achievable and cheap to verify.** Hand-written backprop
  passed a full gradient check at `2.9e-9` relative error, and the network
  reached 98.89% test accuracy with the first hyperparameters tried.
- **Reproducibility is excellent.** Same seeds give byte-identical models across
  runs and across JIT and AOT.
- **AOT gives fast startup and small binaries. Confirmed.** A 6.5 MB
  self-contained server answers its first request about 25 ms after process
  start, roughly 40× faster than `dart run`.
- **One isolate is a bottleneck under concurrent load. Confirmed, with a
  qualifier.** When the model dominates (batch 100), 11 isolates gave 5.3× the
  throughput of one (about 8,000 vs 1,520 req/s) and cut p50 from ~37 ms to
  ~5 ms. That is well short of 11× because six of the cores are efficiency
  cores and the load generator occupies one. At batch 1 the measured gain was
  only 1.65×, but that run was limited by the generator; with two generators
  the 11-isolate server reached 53,000 req/s, 2.5× the single isolate.
- **No GPU, no autograd, thin ecosystem.** All true, and each was felt directly:
  every gradient formula here was derived and written by hand, there is no
  BLAS, and the allowed dependency list has no numeric package on it because
  none is needed to reach this point, but also because there is little to reach
  for beyond it.
- **Code sharing with Flutter.** `lib/` imports only `dart:math` and
  `dart:typed_data`, so the same `Matrix` and `MLP` classes and the same
  `model.json` would run inside a Flutter app unchanged. I did not build or run
  a Flutter app, so this is a property of the code, not a measurement.

### Where the results contradict expectations

- **The i-k-j "cache effect" barely shows up.** Under AOT, i-k-j and the naive
  order are indistinguishable up to n = 256 (0.96–1.03×) and only 1.12–1.18×
  apart at 512. Under JIT the gap is 1.14–1.39×. I expected several-fold. The
  likely reason is that a 512 × 512 Float64 matrix is 2 MB and fits comfortably
  in this CPU's cache, and the prefetcher copes with a fixed column stride. The
  effect should appear at sizes that overflow the cache; I did not measure
  beyond 512, so that is a prediction.
- **Dart matmul is flat at about 2 GFLOPS at every size.** It neither speeds up
  nor slows down between n = 32 and n = 512. That is one multiply-add per
  nanosecond: scalar code with no SIMD and no loop unrolling. A side experiment
  that unrolled the inner loop by hand 4× gained about 30%, so the ceiling for
  scalar Dart on this machine is in the 2–3 GFLOPS range.
- **AOT is not meaningfully faster than JIT at steady state.** Training was only
  about 4% faster under AOT and single-prediction latency was the same. AOT's
  advantage here is startup and deployment, not throughput.
- **Batching does not make inference cheaper per row.** One row at a time costs
  4.7 µs; a batch of 10,000 costs 4.7 µs per row. With NumPy, batching is where
  the speed comes from; in scalar Dart the per-call overhead is already
  negligible, so there is nothing for batching to amortise. Batching still
  matters enormously at the HTTP level: one isolate serves 21,000 rows/s at
  batch 1 and 152,000 rows/s at batch 100.
- **For single predictions the model is not the bottleneck; HTTP is.** One
  isolate handles about 21,000 req/s, which is about 47 µs of CPU per request,
  of which inference is about 6 µs.

### Dart vs NumPy

- **Dart beats NumPy on single predictions. Confirmed.** 4.7–4.9 µs against
  8.4–8.9 µs, about 1.8× faster. NumPy's fixed cost per call (argument checks,
  dispatch, temporary arrays) outweighs the arithmetic for a 1 × 2 input.
- **NumPy/BLAS wins as matrices grow. Confirmed, and by more than I expected.**
  At n = 512 NumPy reaches about 350 GFLOPS against Dart's 2.1, roughly 165×.
  NumPy is already 24× ahead at n = 32, so there is no matmul size in this
  benchmark where Dart wins; the crossover sits below whole-matrix products, at
  the level of a single tiny forward pass.
- **Training is 7.6× faster in NumPy** (about 535 ms against 4,100 ms), even
  though the batches are only 32 rows and the largest product is 64 × 64.
- **Batched inference is 15× faster in NumPy**: 0.31 µs per row against 4.7 µs
  at 10,000 rows. Batching cuts NumPy's per-row cost by 28×; it does nothing for
  Dart.
- **Startup.** `import numpy` alone takes about 50 ms in this environment, twice
  the Dart server's entire cold start. This is an import timing, not a
  comparison against an equivalent Python HTTP server, which I did not build.

### Other weaknesses worth noting

- `model.json` spends about 20 bytes per parameter. It is fine at 4,547
  parameters and would be the wrong format at a few million.
- Every `matmul` allocates its result. The training loop creates several
  short-lived matrices per batch; that did not show up as a problem at this
  size but was not profiled.

## Next experiments

1. **Float32 vs Float64**: halve memory traffic and see whether throughput moves.
2. **SIMD with `Float32x4`**: the most direct attack on the 2 GFLOPS ceiling
   that stays within pure Dart.
3. **Adam optimizer**: fewer epochs to the same accuracy, measured in wall time.
4. **Binary serialization**: raw little-endian `Float64List` bytes instead of JSON.
5. **On-device inference in Flutter**: reuse `lib/` and `model.json` in an app
   and measure latency on a phone.
6. **FFI to BLAS as a contrast**: the same `Matrix` API backed by Accelerate or
   OpenBLAS, to see how much of the gap is the language and how much is the
   library.
7. **Larger matmul sizes (1024, 2048)**: find where the loop-order effect
   actually appears on this CPU.
