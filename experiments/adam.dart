import 'dart:math' as math;

import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';

/// The Adam optimizer (Kingma & Ba, 2015) for an [MLP].
///
/// Keeps a running mean (`m`) and uncentred variance (`v`) of each gradient
/// and scales every parameter's step by `m / sqrt(v)`, with bias correction
/// for the zero-initialised averages. Built on `MLP.computeGradients`, so the
/// model class itself needs no changes.
class Adam {
  final double learningRate;
  final double beta1;
  final double beta2;
  final double epsilon;

  final List<Matrix> _m;
  final List<Matrix> _v;
  int _step = 0;

  Adam(
    MLP model, {
    this.learningRate = 0.001,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
  })  : _m = _zerosLike(model),
        _v = _zerosLike(model);

  // Weights first, then biases; trainBatch walks parameters in this order.
  static List<Matrix> _zerosLike(MLP model) => [
        for (final w in model.weights) Matrix(w.rows, w.cols),
        for (final b in model.biases) Matrix(b.rows, b.cols),
      ];

  /// Runs one Adam step on the batch and returns the loss before the update.
  double trainBatch(MLP model, Matrix x, List<int> labels) {
    final g = model.computeGradients(x, labels);
    _step++;
    // Folding both bias corrections into the step size avoids computing
    // m-hat and v-hat per element.
    final stepSize = learningRate *
        math.sqrt(1 - math.pow(beta2, _step)) /
        (1 - math.pow(beta1, _step));
    final params = [...model.weights, ...model.biases];
    final grads = [...g.dW, ...g.db];
    for (var t = 0; t < params.length; t++) {
      final p = params[t].data, gr = grads[t].data;
      final m = _m[t].data, v = _v[t].data;
      for (var i = 0; i < p.length; i++) {
        final gi = gr[i];
        final mi = m[i] = beta1 * m[i] + (1 - beta1) * gi;
        final vi = v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        p[i] -= stepSize * mi / (math.sqrt(vi) + epsilon);
      }
    }
    return g.loss;
  }
}
