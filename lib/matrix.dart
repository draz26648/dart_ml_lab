import 'dart:math' as math;
import 'dart:typed_data';

/// A dense, row-major matrix of doubles backed by a single [Float64List].
///
/// Element `(r, c)` lives at `data[r * cols + c]`. A `Float64List` stores raw
/// 8-byte doubles contiguously, whereas a `List<double>` may store boxed
/// values, which costs an indirection per element and defeats the cache.
class Matrix {
  final int rows;
  final int cols;
  final Float64List data;

  /// Creates a zero-filled [rows] x [cols] matrix.
  Matrix(this.rows, this.cols) : data = Float64List(rows * cols);

  /// Creates a matrix from [values] given in row-major order.
  Matrix.fromList(this.rows, this.cols, List<double> values)
      : data = Float64List.fromList(values) {
    if (values.length != rows * cols) {
      throw ArgumentError(
        'Expected ${rows * cols} values for a ${rows}x$cols matrix, '
        'got ${values.length}.',
      );
    }
  }

  /// Creates a matrix with entries sampled uniformly from `[-bound, bound]`.
  factory Matrix.uniform(int rows, int cols, math.Random rng, double bound) {
    final m = Matrix(rows, cols);
    final d = m.data;
    for (var i = 0; i < d.length; i++) {
      d[i] = (rng.nextDouble() * 2 - 1) * bound;
    }
    return m;
  }

  double get(int r, int c) => data[r * cols + c];

  void set(int r, int c, double v) {
    data[r * cols + c] = v;
  }

  /// Returns a copy of row [r].
  Float64List row(int r) => data.sublist(r * cols, (r + 1) * cols);

  void _checkMatmulShape(Matrix other) {
    if (cols != other.rows) {
      throw ArgumentError(
        'Cannot multiply ${rows}x$cols by ${other.rows}x${other.cols}.',
      );
    }
  }

  /// Matrix product using i-k-j loop order.
  ///
  /// With `k` in the middle, the innermost loop reads a row of [other] and
  /// writes a row of the output, both sequentially in memory. The classic
  /// i-j-k order instead strides down a column of [other], touching a new
  /// cache line on every step once the matrix outgrows the cache.
  Matrix matmul(Matrix other) {
    _checkMatmulShape(other);
    final n = rows, m = cols, p = other.cols;
    final out = Matrix(n, p);
    final a = data, b = other.data, o = out.data;
    for (var i = 0; i < n; i++) {
      final aRow = i * m;
      final oRow = i * p;
      for (var k = 0; k < m; k++) {
        final aik = a[aRow + k];
        final bRow = k * p;
        for (var j = 0; j < p; j++) {
          o[oRow + j] += aik * b[bRow + j];
        }
      }
    }
    return out;
  }

  /// Matrix product using the classic i-j-k order.
  ///
  /// Only used by the benchmark, to measure the cache effect against [matmul].
  Matrix matmulNaive(Matrix other) {
    _checkMatmulShape(other);
    final n = rows, m = cols, p = other.cols;
    final out = Matrix(n, p);
    final a = data, b = other.data, o = out.data;
    for (var i = 0; i < n; i++) {
      final aRow = i * m;
      final oRow = i * p;
      for (var j = 0; j < p; j++) {
        var sum = 0.0;
        for (var k = 0; k < m; k++) {
          sum += a[aRow + k] * b[k * p + j];
        }
        o[oRow + j] = sum;
      }
    }
    return out;
  }

  Matrix transpose() {
    final out = Matrix(cols, rows);
    final a = data, o = out.data;
    final n = rows, m = cols;
    for (var i = 0; i < n; i++) {
      final aRow = i * m;
      for (var j = 0; j < m; j++) {
        o[j * n + i] = a[aRow + j];
      }
    }
    return out;
  }

  /// Adds the 1 x cols matrix [row] to every row (bias broadcast).
  void addRowInPlace(Matrix row) {
    if (row.rows != 1 || row.cols != cols) {
      throw ArgumentError(
        'Expected a 1x$cols row, got ${row.rows}x${row.cols}.',
      );
    }
    final a = data, b = row.data;
    final n = rows, m = cols;
    for (var i = 0; i < n; i++) {
      final aRow = i * m;
      for (var j = 0; j < m; j++) {
        a[aRow + j] += b[j];
      }
    }
  }

  /// Sums over rows, returning a 1 x cols matrix of column totals.
  Matrix sumRows() {
    final out = Matrix(1, cols);
    final a = data, o = out.data;
    final n = rows, m = cols;
    for (var i = 0; i < n; i++) {
      final aRow = i * m;
      for (var j = 0; j < m; j++) {
        o[j] += a[aRow + j];
      }
    }
    return out;
  }

  /// Computes `this -= other * scale` element-wise.
  void subtractScaledInPlace(Matrix other, double scale) {
    if (other.rows != rows || other.cols != cols) {
      throw ArgumentError(
        'Shape mismatch: ${rows}x$cols vs ${other.rows}x${other.cols}.',
      );
    }
    final a = data, b = other.data;
    for (var i = 0; i < a.length; i++) {
      a[i] -= b[i] * scale;
    }
  }

  void reluInPlace() {
    final a = data;
    for (var i = 0; i < a.length; i++) {
      if (a[i] < 0) a[i] = 0;
    }
  }

  /// Gathers the rows at [indices] into a new matrix, in the given order.
  Matrix rowSlice(List<int> indices) {
    final m = cols;
    final out = Matrix(indices.length, m);
    final a = data, o = out.data;
    for (var i = 0; i < indices.length; i++) {
      final r = indices[i];
      if (r < 0 || r >= rows) {
        throw RangeError.range(r, 0, rows - 1, 'indices[$i]');
      }
      o.setRange(i * m, (i + 1) * m, a, r * m);
    }
    return out;
  }

  /// Replaces each row with its softmax.
  ///
  /// Subtracting the row maximum first makes the largest exponent exactly 0,
  /// so `exp` can never overflow; the result is mathematically unchanged
  /// because the common factor cancels in the normalisation.
  void softmaxRowsInPlace() {
    final a = data;
    final n = rows, m = cols;
    for (var i = 0; i < n; i++) {
      final aRow = i * m;
      var maxV = a[aRow];
      for (var j = 1; j < m; j++) {
        final v = a[aRow + j];
        if (v > maxV) maxV = v;
      }
      var sum = 0.0;
      for (var j = 0; j < m; j++) {
        final e = math.exp(a[aRow + j] - maxV);
        a[aRow + j] = e;
        sum += e;
      }
      for (var j = 0; j < m; j++) {
        a[aRow + j] /= sum;
      }
    }
  }
}
