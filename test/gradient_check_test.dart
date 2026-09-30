import 'dart:math';

import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';
import 'package:test/test.dart';

const _epsilon = 1e-5;
const _tolerance = 1e-5;

double _relativeError(double a, double n) =>
    (a - n).abs() / max(1e-8, a.abs() + n.abs());

/// Compares every entry of [analytic] against the central finite difference
/// of the loss with respect to the matching entry of [param].
///
/// Returns the largest relative error seen.
double _check(
  MLP model,
  Matrix x,
  List<int> labels,
  Matrix param,
  Matrix analytic,
  String name,
) {
  var worst = 0.0;
  for (var i = 0; i < param.data.length; i++) {
    final original = param.data[i];
    param.data[i] = original + _epsilon;
    final lossPlus = model.computeGradients(x, labels).loss;
    param.data[i] = original - _epsilon;
    final lossMinus = model.computeGradients(x, labels).loss;
    param.data[i] = original;

    final numeric = (lossPlus - lossMinus) / (2 * _epsilon);
    final error = _relativeError(analytic.data[i], numeric);
    expect(
      error,
      lessThan(_tolerance),
      reason: '$name[$i]: analytic=${analytic.data[i]} numeric=$numeric',
    );
    worst = max(worst, error);
  }
  return worst;
}

void main() {
  test('analytic gradients match central finite differences', () {
    final rng = Random(11);
    final model = MLP([2, 5, 4, 3], rng);
    // Biases start at zero; randomise them so the check covers a generic
    // point in parameter space rather than a special one.
    for (final b in model.biases) {
      for (var i = 0; i < b.data.length; i++) {
        b.data[i] = (rng.nextDouble() * 2 - 1) * 0.5;
      }
    }
    final x = Matrix.uniform(6, 2, rng, 1);
    final labels = [for (var i = 0; i < 6; i++) rng.nextInt(3)];

    final g = model.computeGradients(x, labels);

    // Guard against a vacuous pass: the check is only meaningful if ReLU is
    // actually switching units on and off and gradients are non-zero.
    final nonZero = [
      for (final m in [...g.dW, ...g.db]) ...m.data,
    ].where((v) => v != 0).length;
    expect(nonZero, greaterThan(model.parameterCount ~/ 2));

    var worst = 0.0;
    var checked = 0;
    for (var l = 0; l < model.layerCount; l++) {
      worst = max(
        worst,
        _check(model, x, labels, model.weights[l], g.dW[l], 'dW[$l]'),
      );
      worst = max(
        worst,
        _check(model, x, labels, model.biases[l], g.db[l], 'db[$l]'),
      );
      checked += model.weights[l].data.length + model.biases[l].data.length;
    }
    expect(checked, model.parameterCount);
    printOnFailure('max relative error: $worst');
    // ignore: avoid_print
    print('gradient check: $checked parameters, max relative error $worst');
  });

  test('computeGradients does not modify the parameters', () {
    final model = MLP([2, 5, 4, 3], Random(1));
    final before = [for (final w in model.weights) w.data.toList()];
    model.computeGradients(Matrix.uniform(4, 2, Random(2), 1), [0, 1, 2, 0]);
    for (var l = 0; l < model.layerCount; l++) {
      expect(model.weights[l].data, before[l]);
    }
  });
}
