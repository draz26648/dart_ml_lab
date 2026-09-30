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
`compare/numpy_baseline.py` is the same experiment in NumPy. `experiments/`
holds the follow-up experiments; nothing in `lib/` or `bin/` depends on it, and
it is the only place that uses `dart:ffi`.

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
  effect should appear at sizes that overflow the cache. The follow-up
  experiments below confirm it: the gap is 1.3× at 1024 and 4.3× at 2048.
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

## Follow-up experiments

All code is in `experiments/`; `lib/`, `bin/` and `test/` are unchanged from the
baseline above. Same machine, same day, same load caveat. AOT numbers are from
two runs unless stated; JIT from one.

```bash
dart test experiments/test                 # 16 tests
dart run experiments/bench_kernels.dart    # 1, 2, 7 (takes about a minute)
dart run experiments/bench_adam.dart       # 3
dart run experiments/bench_binary.dart     # 4 (needs model.json)
dart run experiments/bench_ffi_blas.dart   # 6 (macOS only)
```

### 1, 2, 7: Float32, SIMD and larger sizes

Single-thread GFLOPS, AOT, first run (the second was within 7%):

| n | f64 naive | f64 i-k-j (baseline) | f64x2 SIMD † | f32 scalar | f32x4 SIMD | i-k-j vs naive | f32x4 vs f64 |
|---|---|---|---|---|---|---|---|
| 32 | 2.05 | 2.05 | 3.67 | 1.88 | 6.44 | 1.00× | 3.15× |
| 64 | 2.12 | 2.10 | 3.57 | 1.92 | 7.09 | 0.99× | 3.37× |
| 128 | 2.08 | 2.04 | 3.60 | 1.86 | 7.17 | 0.98× | 3.52× |
| 256 | 2.13 | 2.09 | 3.56 | 1.88 | 7.40 | 0.98× | 3.54× |
| 512 | 1.86 | 2.11 | 3.62 | 1.90 | 7.39 | 1.14× | 3.50× |
| 1024 | 1.66 | 2.11 | 3.64 | 1.91 | 7.29 | 1.28× | 3.45× |
| 2048 | 0.50 | 2.16 | 3.65 | 1.92 | 7.43 | 4.28× | 3.44× |

† Extra, not on the original list.

Under JIT the SIMD kernels are noticeably slower than under AOT: f32x4 reaches
4.4–4.8 GFLOPS and f64x2 2.5–2.6, while the scalar kernels are about the same.

- **Float32 alone does nothing (experiment 1).** Scalar Float32 is 1.9 GFLOPS
  against 2.1 for Float64, about 10% *slower* under AOT, at every size including
  2048. Dart has no single-precision arithmetic: each element is widened to a
  double, multiplied, and narrowed again on the store. Halving memory traffic
  does not help because this loop was never limited by memory.
- **`Float32x4` works (experiment 2).** 7.4 GFLOPS, 3.5× the baseline, in pure
  Dart. It is close to the 4× the lane count suggests and is flat across sizes.
  It costs precision (results agree with Float64 to about 1e-4 in the tests)
  and requires the column count to be a multiple of 4.
- **`Float64x2` gives 1.7× with no loss of precision**; its results are
  bit-identical to `Matrix.matmul`. It needs an even column count, so the
  64-wide hidden layers qualify and the 3-wide output layer does not.
- **The loop-order effect appears at 2048 (experiment 7).** Naive i-j-k falls
  from about 2 GFLOPS to 0.50, while i-k-j holds at 2.16: a 4.3× gap, against
  1.3× at 1024 and nothing at 256 and below. A 2048 × 2048 Float64 matrix is
  32 MB, so this is the point where the operands no longer fit in cache.
  i-k-j is the right default; it just was not needed at the sizes this model
  uses.
- Even the best pure-Dart kernel is still about 45× short of BLAS at n = 512.

I did not wire a SIMD kernel into `MLP`, so there is no end-to-end training or
inference number for it. Training time is matmul-bound, so a gain close to the
kernel's is likely, but that is an inference, not a measurement.

### 3: Adam

Same data, split, init seed, batch size and shuffle seed as `bin/train.dart`.
Targets are checked at the end of each epoch; times exclude evaluation. Both
runs gave identical epochs and losses; times are from the first.

| Optimizer | Loss ≤ 0.05 | Loss ≤ 0.02 | Test ≥ 97% | Final loss (400 ep) | Final test | ms/epoch |
|---|---|---|---|---|---|---|
| SGD, lr 0.2 (baseline) | 28 ep / 285 ms | 123 ep / 1,260 ms | 11 ep / 120 ms | 0.014706 | 98.89% | 10.2 |
| Adam, lr 0.001 | 36 ep / 367 ms | 87 ep / 894 ms | 17 ep / 176 ms | 0.004533 | 98.33% | 10.2 |
| Adam, lr 0.003 | 14 ep / 139 ms | 36 ep / 361 ms | 8 ep / 79 ms | 0.005018 | 98.89% | 10.1 |
| Adam, lr 0.01 | 7 ep / 71 ms | 18 ep / 185 ms | 4 ep / 41 ms | 0.014094 | 98.89% | 10.3 |
| Adam, lr 0.03 | 4 ep / 41 ms | 14 ep / 144 ms | 4 ep / 41 ms | 0.010068 | 98.33% | 10.1 |

- **Adam costs nothing extra per epoch.** The update touches 4,547 parameters;
  the matmuls dominate. So fewer epochs translates directly into less wall time.
- **With a tuned step size Adam is 4–9× faster to a given loss.** At lr 0.01 it
  reaches loss ≤ 0.02 in 185 ms against 1,260 ms for SGD (6.8×).
- **At the paper's default lr of 0.001 Adam is not clearly better.** It is
  slower than SGD to loss ≤ 0.05 and to 97% test accuracy, and faster only to
  the tighter loss target. The SGD baseline at lr 0.2 was already well tuned
  for this problem.
- **Final test accuracy is the same within noise** (98.33–98.89%, a difference
  of one test point). Adam drives training loss lower, but on 180 test points
  that does not show up as better accuracy.

### 4: Binary serialization

In-memory encode and decode, AOT. The format is a 16-byte header, the layer
sizes, then raw little-endian parameters.

| Model | Format | Size | Bytes/param | Encode ms | Decode ms |
|---|---|---|---|---|---|
| Trained, 4,547 params | JSON | 89.9 KB | 20.3 | 1.06 | 0.51 |
| | binary Float64 | 35.6 KB | 8.0 | 0.0016 | 0.025 |
| | binary Float32 † | 17.8 KB | 4.0 | 0.0047 | 0.025 |
| `[784, 512, 512, 10]`, 669,706 params | JSON | 13.6 MB | 20.8 | 193 | 74 |
| | binary Float64 | 5.2 MB | 8.0 | 0.41 | 3.9 |
| | binary Float32 † | 2.6 MB | 4.0 | 0.76 | 4.0 |

† Extra, not on the original list.

- **Binary Float64 is 2.6× smaller, about 20× faster to load and several
  hundred times faster to write**, and lossless: predictions are bit-identical.
- **Float32 halves the size again** and changes predicted probabilities by at
  most 2e-7 on the trained model.
- **For this model it does not matter.** Loading the 90 KB JSON takes half a
  millisecond, against a 25 ms cold start. It starts to matter at the larger
  size, where JSON takes 74 ms to load.
- The binary decoder is slower than it needs to be: it builds the model through
  `MLP.fromJson`, which copies and checks every element, because `lib/` was
  left untouched and has no other public way to construct a model from existing
  parameters. I did not measure how much of the 3.9 ms that accounts for.

### 6: FFI to BLAS

`cblas_dgemm` from Apple Accelerate through `dart:ffi`, with no extra package.
This deliberately breaks the pure-Dart rule. "BLAS + copy" takes and returns
ordinary `Matrix` objects, copying in and out of native memory on each call;
"BLAS" keeps the buffers in native memory. AOT, first run; microseconds per
call.

| n | Pure Dart µs | BLAS + copy µs | BLAS µs | Pure Dart GFLOPS | BLAS + copy GFLOPS | BLAS GFLOPS |
|---|---|---|---|---|---|---|
| 2 | 0.049 | 0.258 | 0.053 | 0.32 | 0.06 | 0.30 |
| 4 | 0.138 | 0.221 | 0.067 | 0.93 | 0.58 | 1.92 |
| 8 | 0.594 | 0.312 | 0.123 | 1.72 | 3.28 | 8.31 |
| 16 | 4.24 | 0.739 | 0.361 | 1.93 | 11.1 | 22.7 |
| 32 | 31.3 | 1.69 | 0.578 | 2.09 | 38.9 | 113 |
| 64 | 250 | 5.00 | 2.20 | 2.09 | 105 | 238 |
| 128 | 2,054 | 24.5 | 13.4 | 2.04 | 171 | 312 |
| 256 | 16,010 | 162 | 95.2 | 2.10 | 207 | 352 |
| 512 | 126,939 | 1,072 | 822 | 2.11 | 250 | 327 |
| 1024 | not run | 7,346 | 6,114 | not run | 292 | 351 |
| 2048 | not run | 56,357 | 51,480 | not run | 305 | 334 |

- **The gap to NumPy is the library, not the language.** Dart calling
  Accelerate reaches 330–350 GFLOPS at 256 and above, the same as NumPy's
  343–357 on the same machine. JIT and AOT give the same figures.
- **FFI overhead is tiny.** A call costs about 50 ns. With native buffers BLAS
  is ahead of pure Dart from n = 4; through the copying `Matrix` API it is ahead
  from n = 8. At n = 2 pure Dart ties the native-buffer call and beats the
  copying one.
- **The copies cost up to half the throughput at mid sizes** (105 against 238
  GFLOPS at n = 64) and under 10% at 2048. A serious FFI-backed `Matrix` would
  keep its data in native memory.
- **It is faster than NumPy at small sizes**: 113 GFLOPS at n = 32 against
  NumPy's 50, since Dart's call path into BLAS is shorter than NumPy's.
- The price is everything the pure-Dart version gave for free: this file is
  macOS-only, manages memory by hand, and would need a different library on
  every other platform, including each Flutter target.

### What the follow-ups change in the verdict

Pure Dart can be pushed from 2 to about 7 GFLOPS with `Float32x4`, and Adam
cuts training time several-fold, so the pure-Dart story is better than the
baseline suggested, but it remains one thread and roughly 45× behind BLAS.
The FFI result is the more important one: with a native BLAS underneath, Dart
matches NumPy at large sizes and beats it at small ones. The honest summary is
that Dart the language is not the obstacle; the missing numeric library is.

## Next experiments

1. **On-device inference in Flutter** (experiment 5 from the original list,
   not done): reuse `lib/` and `model.json` in an app and measure latency on a
   phone.
2. **Wire a SIMD kernel into `MLP`** and measure training and inference end to
   end, rather than inferring it from the kernel benchmark.
3. **Multi-isolate matmul**: split rows across isolates to see how close pure
   Dart gets to BLAS when it is also allowed every core.
4. **A public `MLP.fromParameters` constructor** so the binary decoder can skip
   the `fromJson` copy.
