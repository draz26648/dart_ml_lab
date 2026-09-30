import 'dart:math';

import 'package:dart_ml_lab/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('construction', () {
    test('default constructor is zero-filled', () {
      final m = Matrix(2, 3);
      expect(m.data, everyElement(0.0));
      expect(m.data.length, 6);
    });

    test('fromList throws on length mismatch', () {
      expect(() => Matrix.fromList(2, 2, [1, 2, 3]), throwsArgumentError);
    });

    test('uniform stays within the bound and is seed-deterministic', () {
      final a = Matrix.uniform(10, 10, Random(1), 0.5);
      final b = Matrix.uniform(10, 10, Random(1), 0.5);
      expect(a.data, everyElement(inInclusiveRange(-0.5, 0.5)));
      expect(a.data, b.data);
    });

    test('get, set and row use row-major indexing', () {
      final m = Matrix.fromList(2, 3, [1, 2, 3, 4, 5, 6]);
      expect(m.get(1, 0), 4);
      m.set(0, 2, 9);
      expect(m.row(0), [1, 2, 9]);
      expect(m.row(1), [4, 5, 6]);
    });
  });

  group('matmul', () {
    test('square, hand-computed', () {
      final a = Matrix.fromList(2, 2, [1, 2, 3, 4]);
      final b = Matrix.fromList(2, 2, [5, 6, 7, 8]);
      expect(a.matmul(b).data, [19, 22, 43, 50]);
    });

    test('non-square 2x3 times 3x2, hand-computed', () {
      final a = Matrix.fromList(2, 3, [1, 2, 3, 4, 5, 6]);
      final b = Matrix.fromList(3, 2, [7, 8, 9, 10, 11, 12]);
      final c = a.matmul(b);
      expect((c.rows, c.cols), (2, 2));
      expect(c.data, [58, 64, 139, 154]);
    });

    test('non-square 3x1 times 1x4 (outer product), hand-computed', () {
      final a = Matrix.fromList(3, 1, [1, 2, 3]);
      final b = Matrix.fromList(1, 4, [1, 0, -1, 2]);
      final c = a.matmul(b);
      expect((c.rows, c.cols), (3, 4));
      expect(c.data, [1, 0, -1, 2, 2, 0, -2, 4, 3, 0, -3, 6]);
    });

    test('equals matmulNaive on random matrices', () {
      final rng = Random(3);
      for (final (n, m, p) in [
        (1, 1, 1),
        (4, 7, 5),
        (17, 3, 29),
        (32, 32, 32)
      ]) {
        final a = Matrix.uniform(n, m, rng, 1);
        final b = Matrix.uniform(m, p, rng, 1);
        final fast = a.matmul(b);
        final naive = a.matmulNaive(b);
        expect((fast.rows, fast.cols), (n, p));
        for (var i = 0; i < fast.data.length; i++) {
          expect(fast.data[i], closeTo(naive.data[i], 1e-12));
        }
      }
    });

    test('throws on shape mismatch', () {
      final a = Matrix(2, 3);
      final b = Matrix(2, 3);
      expect(() => a.matmul(b), throwsArgumentError);
      expect(() => a.matmulNaive(b), throwsArgumentError);
    });
  });

  group('transpose', () {
    test('swaps rows and columns', () {
      final t = Matrix.fromList(2, 3, [1, 2, 3, 4, 5, 6]).transpose();
      expect((t.rows, t.cols), (3, 2));
      expect(t.data, [1, 4, 2, 5, 3, 6]);
    });

    test('transposing twice is the identity', () {
      final a = Matrix.uniform(5, 8, Random(4), 1);
      final tt = a.transpose().transpose();
      expect((tt.rows, tt.cols), (5, 8));
      expect(tt.data, a.data);
    });
  });

  group('element-wise helpers', () {
    test('addRowInPlace broadcasts a bias row', () {
      final m = Matrix.fromList(2, 2, [1, 2, 3, 4]);
      m.addRowInPlace(Matrix.fromList(1, 2, [10, 20]));
      expect(m.data, [11, 22, 13, 24]);
    });

    test('addRowInPlace rejects a row of the wrong shape', () {
      expect(
          () => Matrix(2, 2).addRowInPlace(Matrix(1, 3)), throwsArgumentError);
      expect(
          () => Matrix(2, 2).addRowInPlace(Matrix(2, 2)), throwsArgumentError);
    });

    test('sumRows returns 1 x cols column totals', () {
      final s = Matrix.fromList(3, 2, [1, 2, 3, 4, 5, 6]).sumRows();
      expect((s.rows, s.cols), (1, 2));
      expect(s.data, [9, 12]);
    });

    test('subtractScaledInPlace', () {
      final m = Matrix.fromList(1, 3, [1, 2, 3]);
      m.subtractScaledInPlace(Matrix.fromList(1, 3, [2, 4, 6]), 0.5);
      expect(m.data, [0, 0, 0]);
      expect(
        () => m.subtractScaledInPlace(Matrix(3, 1), 1),
        throwsArgumentError,
      );
    });

    test('reluInPlace zeroes negatives only', () {
      final m = Matrix.fromList(1, 4, [-1, 0, 2, -0.5])..reluInPlace();
      expect(m.data, [0, 0, 2, 0]);
    });

    test('rowSlice gathers rows in the requested order', () {
      final m = Matrix.fromList(3, 2, [1, 2, 3, 4, 5, 6]);
      final s = m.rowSlice([2, 0, 2]);
      expect((s.rows, s.cols), (3, 2));
      expect(s.data, [5, 6, 1, 2, 5, 6]);
      expect(() => m.rowSlice([3]), throwsRangeError);
    });
  });

  group('softmaxRowsInPlace', () {
    test('each row sums to 1 and matches hand-computed values', () {
      final m = Matrix.fromList(2, 3, [1, 2, 3, 0, 0, 0])..softmaxRowsInPlace();
      for (var r = 0; r < 2; r++) {
        expect(m.row(r).reduce((a, b) => a + b), closeTo(1, 1e-12));
      }
      final z = exp(1) + exp(2) + exp(3);
      expect(m.get(0, 0), closeTo(exp(1) / z, 1e-12));
      expect(m.get(0, 2), closeTo(exp(3) / z, 1e-12));
      expect(m.row(1), everyElement(closeTo(1 / 3, 1e-12)));
    });

    test('stays finite with large inputs', () {
      final big = Matrix.fromList(1, 3, [1000, 1001, 1002])
        ..softmaxRowsInPlace();
      final small = Matrix.fromList(1, 3, [0, 1, 2])..softmaxRowsInPlace();
      expect(big.data.every((v) => v.isFinite), isTrue);
      expect(big.data.reduce((a, b) => a + b), closeTo(1, 1e-12));
      // Softmax is shift-invariant, so the answer must match the small case.
      for (var j = 0; j < 3; j++) {
        expect(big.data[j], closeTo(small.data[j], 1e-12));
      }
    });

    test('stays finite with very negative inputs', () {
      final m = Matrix.fromList(1, 2, [-1000, -1001])..softmaxRowsInPlace();
      expect(m.data.every((v) => v.isFinite), isTrue);
      expect(m.data.reduce((a, b) => a + b), closeTo(1, 1e-12));
    });
  });
}
