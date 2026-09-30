// Experiments 1, 2 and 7: Float32 vs Float64, Float32x4 SIMD, and the
// loop-order effect at sizes beyond 512. Plus one extra: Float64x2 SIMD.
//
//   dart run experiments/bench_kernels.dart
//   dart compile exe experiments/bench_kernels.dart -o build/bench_kernels

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:dart_ml_lab/matrix.dart';

import 'kernels.dart';
import 'timing.dart';

const sizes = [32, 64, 128, 256, 512, 1024, 2048];
const benchSeed = 123;

void main() {
  var sink = 0.0;
  print('matmul kernels (Dart ${Platform.version.split(' ').first}), '
      'square n x n, single thread, GFLOPS = 2n^3 / time');
  print('${'n'.padLeft(5)}  ${'f64 naive'.padLeft(10)}  '
      '${'f64 ikj'.padLeft(10)}  ${'f64x2 simd'.padLeft(10)}  '
      '${'f32 ikj'.padLeft(10)}  ${'f32x4 simd'.padLeft(10)}  '
      '${'ikj/naive'.padLeft(9)}  ${'f32x4/f64'.padLeft(9)}');
  final rng = Random(benchSeed);
  for (final n in sizes) {
    final a = Matrix.uniform(n, n, rng, 1);
    final b = Matrix.uniform(n, n, rng, 1);
    final a32 = Float32List.fromList(a.data);
    final b32 = Float32List.fromList(b.data);

    final results = [
      measureMs(() => sink += a.matmulNaive(b).data[n]),
      measureMs(() => sink += a.matmul(b).data[n]),
      measureMs(() => sink += matmulF64x2(a.data, b.data, n, n, n)[n]),
      measureMs(() => sink += matmulF32(a32, b32, n, n, n)[n]),
      measureMs(() => sink += matmulF32x4(a32, b32, n, n, n)[n]),
    ];
    final g = [for (final ms in results) gflops(n, ms)];
    print('${n.toString().padLeft(5)}  '
        '${[for (final v in g) v.toStringAsFixed(2).padLeft(10)].join('  ')}  '
        '${'${(g[1] / g[0]).toStringAsFixed(2)}x'.padLeft(9)}  '
        '${'${(g[4] / g[1]).toStringAsFixed(2)}x'.padLeft(9)}');
  }
  print('sink: ${sink.toStringAsFixed(3)}');
}
