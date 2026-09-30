import 'dart:io';
import 'dart:math';

import 'package:dart_ml_lab/matrix.dart';

const sizes = [32, 64, 128, 256, 512];
const benchSeed = 123;
const minMeasureMicros = 200 * 1000;
const warmupMicros = 100 * 1000;

typedef Multiply = Matrix Function(Matrix a, Matrix b);

/// Accumulates one element per product so the work can't be optimised away.
double sink = 0;

/// Runs [multiply] repeatedly for at least [micros]; returns (reps, elapsed).
(int, int) runFor(Multiply multiply, Matrix a, Matrix b, int micros) {
  final watch = Stopwatch()..start();
  var reps = 0;
  while (watch.elapsedMicroseconds < micros) {
    sink += multiply(a, b).data[reps % a.data.length];
    reps++;
  }
  return (reps, watch.elapsedMicroseconds);
}

/// Returns the mean milliseconds per multiplication.
double measure(Multiply multiply, Matrix a, Matrix b) {
  runFor(multiply, a, b, warmupMicros);
  // Fix the repetition count from a calibration pass, then time exactly that
  // many calls with no clock reads inside the loop.
  final (calibReps, calibMicros) = runFor(multiply, a, b, minMeasureMicros);
  final reps = max(1, (calibReps * minMeasureMicros / calibMicros).ceil());
  final watch = Stopwatch()..start();
  for (var i = 0; i < reps; i++) {
    sink += multiply(a, b).data[i % a.data.length];
  }
  watch.stop();
  return watch.elapsedMicroseconds / 1000 / reps;
}

double gflops(int n, double ms) => 2.0 * n * n * n / (ms / 1000) / 1e9;

void main() {
  print('matmul benchmark (Dart ${Platform.version.split(' ').first}), '
      'square n x n, Float64');
  print('${'n'.padLeft(5)}  ${'naive ms'.padLeft(11)}  '
      '${'naive GFLOPS'.padLeft(12)}  ${'ikj ms'.padLeft(11)}  '
      '${'ikj GFLOPS'.padLeft(10)}  ${'speedup'.padLeft(7)}');
  final rng = Random(benchSeed);
  for (final n in sizes) {
    final a = Matrix.uniform(n, n, rng, 1);
    final b = Matrix.uniform(n, n, rng, 1);
    final naiveMs = measure((x, y) => x.matmulNaive(y), a, b);
    final ikjMs = measure((x, y) => x.matmul(y), a, b);
    print('${n.toString().padLeft(5)}  '
        '${naiveMs.toStringAsFixed(4).padLeft(11)}  '
        '${gflops(n, naiveMs).toStringAsFixed(3).padLeft(12)}  '
        '${ikjMs.toStringAsFixed(4).padLeft(11)}  '
        '${gflops(n, ikjMs).toStringAsFixed(3).padLeft(10)}  '
        '${'${(naiveMs / ikjMs).toStringAsFixed(2)}x'.padLeft(7)}');
  }
  print('sink: ${sink.toStringAsFixed(6)}');
}
