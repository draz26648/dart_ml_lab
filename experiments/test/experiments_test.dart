import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:dart_ml_lab/dataset.dart';
import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';
import 'package:test/test.dart';

import '../adam.dart';
import '../binary_format.dart';
import '../blas_ffi.dart';
import '../kernels.dart';

void main() {
  group('kernels match Matrix.matmul', () {
    final rng = Random(3);
    for (final (n, m, p) in [(1, 1, 4), (5, 7, 8), (16, 3, 12), (32, 32, 32)]) {
      final a = Matrix.uniform(n, m, rng, 1);
      final b = Matrix.uniform(m, p, rng, 1);
      final expected = a.matmul(b).data;
      final a32 = Float32List.fromList(a.data);
      final b32 = Float32List.fromList(b.data);

      test('f64x2 is bit-identical at ${n}x${m}x$p', () {
        expect(matmulF64x2(a.data, b.data, n, m, p), expected);
      });
      test('f32 scalar and f32x4 agree with f64 at ${n}x${m}x$p', () {
        final scalar = matmulF32(a32, b32, n, m, p);
        final simd = matmulF32x4(a32, b32, n, m, p);
        for (var i = 0; i < expected.length; i++) {
          expect(scalar[i], closeTo(expected[i], 1e-4));
          expect(simd[i], closeTo(expected[i], 1e-4));
        }
      });
    }

    test('SIMD kernels reject unaligned column counts', () {
      expect(
        () => matmulF32x4(Float32List(6), Float32List(9), 2, 3, 3),
        throwsArgumentError,
      );
      expect(
        () => matmulF64x2(Float64List(6), Float64List(9), 2, 3, 3),
        throwsArgumentError,
      );
      expect(
        () => matmulF32(Float32List(5), Float32List(9), 2, 3, 3),
        throwsArgumentError,
      );
    });
  });

  group('Adam', () {
    test('first step moves every parameter by about the learning rate', () {
      // With bias correction, step one is lr * g / (|g| + eps) = lr * sign(g).
      final model = MLP([2, 4, 3], Random(1));
      final x = Matrix.uniform(6, 2, Random(2), 1);
      final labels = [0, 1, 2, 0, 1, 2];
      final before = model.weights.last.data.toList();
      final g = model.computeGradients(x, labels);
      Adam(model, learningRate: 0.01).trainBatch(model, x, labels);
      for (var i = 0; i < before.length; i++) {
        final grad = g.dW.last.data[i];
        if (grad.abs() < 1e-6) continue;
        expect(
          model.weights.last.data[i] - before[i],
          closeTo(-0.01 * grad.sign, 1e-6),
        );
      }
    });

    test('reduces the loss and is deterministic', () {
      List<double> run() {
        final model = MLP([2, 16, 3], Random(42));
        final d = makeSpiral(
            pointsPerClass: 20, classes: 3, noise: 0.1, rng: Random(7));
        final adam = Adam(model, learningRate: 0.01);
        return [for (var i = 0; i < 60; i++) adam.trainBatch(model, d.x, d.y)];
      }

      final losses = run();
      expect(losses.last, lessThan(losses.first * 0.8));
      expect(run(), losses);
    });
  });

  group('binary format', () {
    final model = MLP([2, 16, 8, 3], Random(42));
    final x = Matrix.uniform(12, 2, Random(5), 1);

    test('float64 round-trip gives bit-identical predictions', () {
      final bytes = encodeModel(model);
      expect(bytes.length, 32 + model.parameterCount * 8);
      final restored = decodeModel(bytes);
      expect(restored.sizes, model.sizes);
      expect(restored.predictProba(x).data, model.predictProba(x).data);
    });

    test('float32 round-trip is half the size and close', () {
      final bytes = encodeModel(model, float32: true);
      expect(bytes.length, 32 + model.parameterCount * 4);
      final p = decodeModel(bytes).predictProba(x).data;
      final q = model.predictProba(x).data;
      for (var i = 0; i < p.length; i++) {
        expect(p[i], closeTo(q[i], 1e-5));
      }
    });

    test('decodes from a view at a non-zero buffer offset', () {
      final bytes = encodeModel(model);
      final shifted = Uint8List(bytes.length + 3)..setAll(3, bytes);
      final restored = decodeModel(Uint8List.sublistView(shifted, 3));
      expect(restored.predictProba(x).data, model.predictProba(x).data);
    });

    test('rejects corrupt input', () {
      final good = encodeModel(model);
      expect(() => decodeModel(Uint8List(4)), throwsFormatException);
      expect(
        () => decodeModel(Uint8List.fromList(good)..[0] = 0),
        throwsFormatException,
      );
      expect(
        () => decodeModel(Uint8List.fromList(good)..[4] = 9),
        throwsFormatException,
      );
      expect(
        () => decodeModel(Uint8List.fromList(good)..[8] = 3),
        throwsFormatException,
      );
      expect(
        () => decodeModel(Uint8List.sublistView(good, 0, good.length - 8)),
        throwsFormatException,
      );
    });
  });

  group('BLAS via FFI', () {
    test('matches Matrix.matmul, including non-square shapes', () {
      final blas = Blas.open();
      final rng = Random(9);
      for (final (n, m, p) in [
        (1, 1, 1),
        (4, 7, 5),
        (17, 3, 29),
        (64, 64, 64)
      ]) {
        final a = Matrix.uniform(n, m, rng, 1);
        final b = Matrix.uniform(m, p, rng, 1);
        final expected = a.matmul(b);
        final actual = blas.matmul(a, b);
        expect((actual.rows, actual.cols), (n, p));
        for (var i = 0; i < expected.data.length; i++) {
          expect(actual.data[i], closeTo(expected.data[i], 1e-10));
        }
      }
      expect(
          () => blas.matmul(Matrix(2, 3), Matrix(2, 3)), throwsArgumentError);
    }, skip: Platform.isMacOS ? null : 'Accelerate is macOS only');
  });
}
