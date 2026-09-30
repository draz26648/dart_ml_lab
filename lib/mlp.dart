import 'dart:math' as math;

import 'matrix.dart';

/// Loss and parameter gradients for one batch, as returned by
/// [MLP.computeGradients].
typedef Gradients = ({double loss, List<Matrix> dW, List<Matrix> db});

/// A fully connected network: Dense -> ReLU -> ... -> Dense -> Softmax.
///
/// Works on whole batches: each row of an input matrix is one sample, so a
/// layer is one `matmul` rather than a graph of scalar operations.
class MLP {
  static const int formatVersion = 1;

  /// Layer widths, e.g. `[2, 64, 64, 3]`.
  final List<int> sizes;

  /// `weights[l]` is `sizes[l] x sizes[l + 1]` (fan_in x fan_out).
  final List<Matrix> weights;

  /// `biases[l]` is `1 x sizes[l + 1]`.
  final List<Matrix> biases;

  MLP._(this.sizes, this.weights, this.biases);

  /// Builds a network with He-uniform weights and zero biases.
  ///
  /// He initialisation keeps the activation variance stable through ReLU
  /// layers: `Var(w) = 2 / fan_in`, which for a uniform distribution means a
  /// bound of `sqrt(6 / fan_in)`.
  factory MLP(List<int> sizes, math.Random rng) {
    if (sizes.length < 2 || sizes.any((s) => s <= 0)) {
      throw ArgumentError('sizes needs at least two positive entries: $sizes');
    }
    final weights = <Matrix>[];
    final biases = <Matrix>[];
    for (var l = 0; l < sizes.length - 1; l++) {
      final fanIn = sizes[l], fanOut = sizes[l + 1];
      weights.add(Matrix.uniform(fanIn, fanOut, rng, math.sqrt(6 / fanIn)));
      biases.add(Matrix(1, fanOut));
    }
    return MLP._(List.unmodifiable(sizes), weights, biases);
  }

  int get layerCount => weights.length;

  int get parameterCount {
    var n = 0;
    for (var l = 0; l < layerCount; l++) {
      n += weights[l].data.length + biases[l].data.length;
    }
    return n;
  }

  /// Runs the forward pass and returns every activation: `[x, a1, ..., probs]`.
  List<Matrix> _forward(Matrix x) {
    if (x.cols != sizes.first) {
      throw ArgumentError('Expected ${sizes.first} features, got ${x.cols}.');
    }
    final activations = <Matrix>[x];
    var a = x;
    for (var l = 0; l < layerCount; l++) {
      a = a.matmul(weights[l])..addRowInPlace(biases[l]);
      if (l < layerCount - 1) {
        a.reluInPlace();
      } else {
        a.softmaxRowsInPlace();
      }
      activations.add(a);
    }
    return activations;
  }

  /// Returns an `x.rows x sizes.last` matrix of class probabilities.
  Matrix predictProba(Matrix x) => _forward(x).last;

  /// Returns the most probable class for each row of [x].
  List<int> predictClasses(Matrix x) => argmaxRows(predictProba(x));

  /// Computes the mean cross-entropy loss and its gradients without changing
  /// any parameter.
  Gradients computeGradients(Matrix x, List<int> labels) {
    if (labels.length != x.rows) {
      throw ArgumentError('Got ${labels.length} labels for ${x.rows} rows.');
    }
    final batch = x.rows;
    final classes = sizes.last;
    final activations = _forward(x);
    final probs = activations.last;

    // Softmax + cross-entropy collapse to a simple gradient with respect to
    // the logits: (probs - one_hot) / batchSize.
    var delta = Matrix(batch, classes);
    final p = probs.data, d = delta.data;
    var loss = 0.0;
    for (var i = 0; i < batch; i++) {
      final label = labels[i];
      if (label < 0 || label >= classes) {
        throw ArgumentError('Label $label is outside 0..${classes - 1}.');
      }
      final rowStart = i * classes;
      // Clamp so a confidently wrong prediction gives a large loss, not inf.
      loss -= math.log(math.max(p[rowStart + label], 1e-12));
      for (var j = 0; j < classes; j++) {
        d[rowStart + j] = p[rowStart + j] / batch;
      }
      d[rowStart + label] -= 1 / batch;
    }
    loss /= batch;

    final dW = List<Matrix?>.filled(layerCount, null);
    final db = List<Matrix?>.filled(layerCount, null);
    for (var l = layerCount - 1; l >= 0; l--) {
      final aPrev = activations[l];
      dW[l] = aPrev.transpose().matmul(delta);
      db[l] = delta.sumRows();
      if (l > 0) {
        // Ordering matters: delta_prev must be computed from the weights that
        // produced this forward pass. If W_l were updated first, the error
        // would be propagated through weights the forward pass never used and
        // every earlier layer would get a gradient for a different network.
        // Here nothing is updated until all gradients exist (see trainBatch),
        // which keeps that guarantee structural rather than a matter of care.
        final deltaPrev = delta.matmul(weights[l].transpose());
        // ReLU derivative: aPrev is the post-ReLU output of layer l - 1, which
        // is <= 0 exactly where its pre-activation was <= 0.
        final dp = deltaPrev.data, ap = aPrev.data;
        for (var i = 0; i < dp.length; i++) {
          if (ap[i] <= 0) dp[i] = 0;
        }
        delta = deltaPrev;
      }
    }
    return (loss: loss, dW: dW.cast<Matrix>(), db: db.cast<Matrix>());
  }

  /// Runs one SGD step on the batch and returns the loss before the update.
  double trainBatch(Matrix x, List<int> labels, double lr) {
    final g = computeGradients(x, labels);
    for (var l = 0; l < layerCount; l++) {
      weights[l].subtractScaledInPlace(g.dW[l], lr);
      biases[l].subtractScaledInPlace(g.db[l], lr);
    }
    return g.loss;
  }

  Map<String, dynamic> toJson() => {
        'format_version': formatVersion,
        'sizes': sizes,
        'weights': [for (final w in weights) w.data.toList()],
        'biases': [for (final b in biases) b.data.toList()],
      };

  /// Rebuilds a model from [toJson] output, validating every shape.
  factory MLP.fromJson(Map<String, dynamic> json) {
    final version = json['format_version'];
    if (version != formatVersion) {
      throw FormatException(
        'Unsupported format_version: $version (expected $formatVersion).',
      );
    }
    final rawSizes = json['sizes'];
    if (rawSizes is! List ||
        rawSizes.length < 2 ||
        rawSizes.any((s) => s is! int || s <= 0)) {
      throw const FormatException(
        '"sizes" must be a list of at least two positive integers.',
      );
    }
    final sizes = List<int>.unmodifiable(rawSizes.cast<int>());
    final layers = sizes.length - 1;

    List<Matrix> parse(String key, int Function(int l) rowsOf) {
      final raw = json[key];
      if (raw is! List || raw.length != layers) {
        throw FormatException(
          '"$key" must be a list of $layers layers to match sizes $sizes.',
        );
      }
      return [
        for (var l = 0; l < layers; l++)
          _parseMatrix(raw[l], rowsOf(l), sizes[l + 1], '$key[$l]'),
      ];
    }

    return MLP._(
      sizes,
      parse('weights', (l) => sizes[l]),
      parse('biases', (_) => 1),
    );
  }

  static Matrix _parseMatrix(Object? raw, int rows, int cols, String name) {
    if (raw is! List || raw.length != rows * cols) {
      final got = raw is List ? '${raw.length} values' : '${raw.runtimeType}';
      throw FormatException(
        '$name must hold ${rows * cols} numbers (${rows}x$cols), got $got.',
      );
    }
    final m = Matrix(rows, cols);
    for (var i = 0; i < raw.length; i++) {
      final v = raw[i];
      // jsonDecode yields an int for values like `1`, so convert explicitly.
      if (v is! num || !v.isFinite) {
        throw FormatException('$name[$i] is not a finite number: $v');
      }
      m.data[i] = v.toDouble();
    }
    return m;
  }
}

/// Returns the column index of the largest value in each row of [m].
List<int> argmaxRows(Matrix m) {
  final d = m.data;
  final cols = m.cols;
  final out = List<int>.filled(m.rows, 0);
  for (var i = 0; i < m.rows; i++) {
    final rowStart = i * cols;
    var best = 0;
    var bestV = d[rowStart];
    for (var j = 1; j < cols; j++) {
      final v = d[rowStart + j];
      if (v > bestV) {
        bestV = v;
        best = j;
      }
    }
    out[i] = best;
  }
  return out;
}
