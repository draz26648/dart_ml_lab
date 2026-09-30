// Experiment 6: FFI to BLAS (Apple Accelerate) as a contrast to pure Dart.
// Breaks the pure-Dart rule on purpose; see blas_ffi.dart. macOS only.
//
//   dart run experiments/bench_ffi_blas.dart
//   dart compile exe experiments/bench_ffi_blas.dart -o build/bench_ffi_blas

import 'dart:io';
import 'dart:math';

import 'package:dart_ml_lab/matrix.dart';

import 'blas_ffi.dart';
import 'timing.dart';

const sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
const pureDartLimit = 512;
const benchSeed = 123;

String us(double ms) => (ms * 1000).toStringAsFixed(3).padLeft(12);

void main() {
  if (!Platform.isMacOS) {
    stderr.writeln('This experiment uses Accelerate and only runs on macOS.');
    exit(1);
  }
  final blas = Blas.open();
  var sink = 0.0;
  print('matmul via FFI to Accelerate vs pure Dart '
      '(Dart ${Platform.version.split(' ').first}), square n x n, Float64');
  print('${'n'.padLeft(5)}  ${'pure Dart us'.padLeft(12)}  '
      '${'BLAS+copy us'.padLeft(12)}  ${'BLAS us'.padLeft(12)}  '
      '${'Dart GFLOPS'.padLeft(11)}  ${'BLAS+copy'.padLeft(10)}  '
      '${'BLAS GFLOPS'.padLeft(11)}');
  final rng = Random(benchSeed);
  for (final n in sizes) {
    final a = Matrix.uniform(n, n, rng, 1);
    final b = Matrix.uniform(n, n, rng, 1);

    // Pure Dart i-k-j, skipped at sizes where it takes seconds per call.
    final dartMs = n <= pureDartLimit
        ? measureMs(() => sink += a.matmul(b).data[n])
        : double.nan;

    // The Matrix-in, Matrix-out API: copies in and out of native memory.
    final copyMs = measureMs(() => sink += blas.matmul(a, b).data[n]);

    // Buffers that live in native memory throughout: the library alone.
    final na = blas.allocate(n * n)..list.setAll(0, a.data);
    final nb = blas.allocate(n * n)..list.setAll(0, b.data);
    final nc = blas.allocate(n * n);
    final nativeMs = measureMs(() {
      blas.dgemm(na, nb, nc, n, n, n);
      sink += nc.list[n];
    });
    blas
      ..free(na)
      ..free(nb)
      ..free(nc);

    print('${n.toString().padLeft(5)}  '
        '${dartMs.isNaN ? '-'.padLeft(12) : us(dartMs)}  '
        '${us(copyMs)}  ${us(nativeMs)}  '
        '${(dartMs.isNaN ? '-' : gflops(n, dartMs).toStringAsFixed(2)).padLeft(11)}  '
        '${gflops(n, copyMs).toStringAsFixed(2).padLeft(10)}  '
        '${gflops(n, nativeMs).toStringAsFixed(2).padLeft(11)}');
  }
  print('sink: ${sink.toStringAsFixed(3)}');
}
