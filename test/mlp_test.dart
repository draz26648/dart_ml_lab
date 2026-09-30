import 'dart:convert';
import 'dart:math';

import 'package:dart_ml_lab/dataset.dart';
import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';
import 'package:test/test.dart';

void main() {
  group('construction', () {
    test('shapes, zero biases and parameter count', () {
      final model = MLP([2, 64, 64, 3], Random(42));
      expect(model.weights.map((w) => (w.rows, w.cols)), [
        (2, 64),
        (64, 64),
        (64, 3),
      ]);
      expect(model.biases.map((b) => (b.rows, b.cols)), [
        (1, 64),
        (1, 64),
        (1, 3),
      ]);
      for (final b in model.biases) {
        expect(b.data, everyElement(0.0));
      }
      expect(model.parameterCount, 2 * 64 + 64 + 64 * 64 + 64 + 64 * 3 + 3);
    });

    test('He init stays within sqrt(6 / fan_in)', () {
      final model = MLP([2, 64, 3], Random(42));
      expect(
        model.weights[0].data.map((v) => v.abs()).reduce(max),
        allOf(lessThanOrEqualTo(sqrt(6 / 2)), greaterThan(sqrt(6 / 2) * 0.9)),
      );
      expect(
        model.weights[1].data.map((v) => v.abs()).reduce(max),
        lessThanOrEqualTo(sqrt(6 / 64)),
      );
    });

    test('same seed gives identical outputs, different seed does not', () {
      final x = Matrix.uniform(8, 2, Random(5), 1);
      final a = MLP([2, 16, 3], Random(42)).predictProba(x);
      final b = MLP([2, 16, 3], Random(42)).predictProba(x);
      final c = MLP([2, 16, 3], Random(43)).predictProba(x);
      expect(a.data, b.data);
      expect(a.data, isNot(c.data));
    });
  });

  group('prediction', () {
    test('probabilities are a distribution and classes are their argmax', () {
      final model = MLP([2, 16, 3], Random(42));
      final x = Matrix.uniform(10, 2, Random(5), 1);
      final p = model.predictProba(x);
      expect((p.rows, p.cols), (10, 3));
      final classes = model.predictClasses(x);
      for (var r = 0; r < 10; r++) {
        final row = p.row(r);
        expect(row.reduce((a, b) => a + b), closeTo(1, 1e-12));
        expect(row[classes[r]], row.reduce(max));
      }
    });

    test('rejects input with the wrong feature count', () {
      final model = MLP([2, 16, 3], Random(42));
      expect(() => model.predictProba(Matrix(1, 3)), throwsArgumentError);
    });
  });

  group('training', () {
    test('trainBatch returns the pre-update loss and then reduces it', () {
      final model = MLP([2, 16, 3], Random(42));
      final data = makeSpiral(
          pointsPerClass: 20, classes: 3, noise: 0.1, rng: Random(7));
      final before = model.computeGradients(data.x, data.y).loss;
      final first = model.trainBatch(data.x, data.y, 0.1);
      expect(first, before);
      var last = first;
      for (var i = 0; i < 50; i++) {
        last = model.trainBatch(data.x, data.y, 0.1);
      }
      expect(last, lessThan(first));
    });

    test('two identically seeded training runs are bit-identical', () {
      List<double> run() {
        final model = MLP([2, 16, 3], Random(42));
        final data = makeSpiral(
            pointsPerClass: 20, classes: 3, noise: 0.1, rng: Random(7));
        return [
          for (var i = 0; i < 20; i++) model.trainBatch(data.x, data.y, 0.1),
        ];
      }

      expect(run(), run());
    });
  });

  group('JSON', () {
    test('round-trip through an encoded string gives identical predictions',
        () {
      final model = MLP([2, 16, 8, 3], Random(42));
      final x = Matrix.uniform(12, 2, Random(5), 1);
      final json = jsonDecode(jsonEncode(model.toJson()));
      final restored = MLP.fromJson(json as Map<String, dynamic>);
      expect(restored.sizes, model.sizes);
      expect(restored.parameterCount, model.parameterCount);
      expect(restored.predictProba(x).data, model.predictProba(x).data);
      expect(json['format_version'], 1);
    });

    test('fromJson converts integer-valued numbers to double', () {
      final model = MLP.fromJson({
        'format_version': 1,
        'sizes': [1, 2],
        'weights': [
          [1, 2]
        ],
        'biases': [
          [0, 0]
        ],
      });
      expect(model.weights[0].data, [1.0, 2.0]);
    });

    group('fromJson rejects', () {
      Map<String, dynamic> valid() =>
          jsonDecode(jsonEncode(MLP([2, 4, 3], Random(1)).toJson()))
              as Map<String, dynamic>;

      void rejects(String name, void Function(Map<String, dynamic>) corrupt) {
        test(name, () {
          final json = valid();
          corrupt(json);
          expect(() => MLP.fromJson(json), throwsFormatException);
        });
      }

      test('nothing when the JSON is valid', () {
        expect(MLP.fromJson(valid()).sizes, [2, 4, 3]);
      });
      rejects('a missing format_version', (j) => j.remove('format_version'));
      rejects('an unknown format_version', (j) => j['format_version'] = 2);
      rejects('missing sizes', (j) => j.remove('sizes'));
      rejects('non-integer sizes', (j) => j['sizes'] = [2, 4.5, 3]);
      rejects('a single-entry sizes', (j) => j['sizes'] = [2]);
      rejects('sizes that disagree with the weights',
          (j) => j['sizes'] = [2, 5, 3]);
      rejects(
          'a missing weight layer', (j) => (j['weights'] as List).removeLast());
      rejects('a weight layer of the wrong length',
          (j) => ((j['weights'] as List)[0] as List).removeLast());
      rejects('a bias layer of the wrong length',
          (j) => ((j['biases'] as List)[1] as List).add(0.0));
      rejects('missing biases', (j) => j.remove('biases'));
      rejects('a non-numeric weight',
          (j) => ((j['weights'] as List)[0] as List)[0] = 'x');
      rejects('a weight layer that is not a list',
          (j) => (j['weights'] as List)[0] = 'oops');
    });
  });

  group('dataset', () {
    test('makeSpiral has the expected shape, labels and radius', () {
      final d = makeSpiral(
          pointsPerClass: 50, classes: 3, noise: 0.2, rng: Random(7));
      expect((d.x.rows, d.x.cols), (150, 2));
      expect(d.y.length, 150);
      for (var c = 0; c < 3; c++) {
        expect(d.y.where((v) => v == c).length, 50);
      }
      // Point i of each class sits at radius i / (pointsPerClass - 1).
      expect(sqrt(pow(d.x.get(0, 0), 2) + pow(d.x.get(0, 1), 2)), 0);
      expect(
        sqrt(pow(d.x.get(49, 0), 2) + pow(d.x.get(49, 1), 2)),
        closeTo(1, 1e-12),
      );
    });

    test('makeSpiral is seed-deterministic', () {
      final a = makeSpiral(
          pointsPerClass: 50, classes: 3, noise: 0.2, rng: Random(7));
      final b = makeSpiral(
          pointsPerClass: 50, classes: 3, noise: 0.2, rng: Random(7));
      expect(a.x.data, b.x.data);
    });

    test('gaussian has roughly zero mean and unit variance', () {
      final rng = Random(9);
      const n = 200000;
      var sum = 0.0, sumSq = 0.0;
      for (var i = 0; i < n; i++) {
        final g = gaussian(rng);
        sum += g;
        sumSq += g * g;
      }
      final mean = sum / n;
      expect(mean, closeTo(0, 0.01));
      expect(sumSq / n - mean * mean, closeTo(1, 0.02));
    });
  });
}
