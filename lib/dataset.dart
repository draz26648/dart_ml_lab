import 'dart:math' as math;

import 'matrix.dart';

/// Draws one standard-normal sample from [rng] using the Box-Muller transform.
double gaussian(math.Random rng) {
  // nextDouble() is in [0, 1); flip it to (0, 1] so log() never sees zero.
  final u1 = 1.0 - rng.nextDouble();
  final u2 = rng.nextDouble();
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
}

/// Generates the interleaved-spirals classification dataset.
///
/// Rows are ordered by class, then by radius; callers shuffle if they need to.
({Matrix x, List<int> y}) makeSpiral({
  required int pointsPerClass,
  required int classes,
  required double noise,
  required math.Random rng,
}) {
  final x = Matrix(pointsPerClass * classes, 2);
  final y = List<int>.filled(pointsPerClass * classes, 0);
  for (var c = 0; c < classes; c++) {
    for (var i = 0; i < pointsPerClass; i++) {
      final r = i / (pointsPerClass - 1);
      final t = c * 4 + r * 4 + gaussian(rng) * noise;
      final idx = c * pointsPerClass + i;
      x.set(idx, 0, r * math.sin(t));
      x.set(idx, 1, r * math.cos(t));
      y[idx] = c;
    }
  }
  return (x: x, y: y);
}
