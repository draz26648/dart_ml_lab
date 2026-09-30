/// Alternative matmul kernels for experiments 1 and 2.
///
/// These are free functions over raw typed lists rather than more `Matrix`
/// variants: the experiment is about the inner loop, and the baseline `Matrix`
/// class stays untouched so the earlier results remain valid.
///
/// Every kernel computes the `n x p` product of a row-major `n x m` matrix [a]
/// and a row-major `m x p` matrix [b], in i-k-j order.
library;

import 'dart:typed_data';

void _checkShapes(int aLength, int bLength, int n, int m, int p) {
  if (aLength != n * m || bLength != m * p) {
    throw ArgumentError(
      'Expected ${n * m} and ${m * p} elements, got $aLength and $bLength.',
    );
  }
}

/// Scalar Float32: the same loop as `Matrix.matmul` over 4-byte floats.
Float32List matmulF32(Float32List a, Float32List b, int n, int m, int p) {
  _checkShapes(a.length, b.length, n, m, p);
  final o = Float32List(n * p);
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
  return o;
}

/// Float32 with 4-lane SIMD: each inner step handles four columns.
///
/// Requires `p % 4 == 0` so every row starts on a vector boundary.
Float32List matmulF32x4(Float32List a, Float32List b, int n, int m, int p) {
  _checkShapes(a.length, b.length, n, m, p);
  if (p % 4 != 0) {
    throw ArgumentError('matmulF32x4 needs a column count divisible by 4.');
  }
  final o = Float32List(n * p);
  // Views over the same bytes: element v covers floats 4v .. 4v + 3.
  final b4 = Float32x4List.view(b.buffer, b.offsetInBytes, b.length ~/ 4);
  final o4 = Float32x4List.view(o.buffer, 0, o.length ~/ 4);
  final p4 = p ~/ 4;
  for (var i = 0; i < n; i++) {
    final aRow = i * m;
    final oRow = i * p4;
    for (var k = 0; k < m; k++) {
      final aik = Float32x4.splat(a[aRow + k]);
      final bRow = k * p4;
      for (var j = 0; j < p4; j++) {
        o4[oRow + j] += b4[bRow + j] * aik;
      }
    }
  }
  return o;
}

/// Extra, not on the original list: Float64 with 2-lane SIMD.
///
/// Keeps full double precision, so it could replace `Matrix.matmul` without
/// changing any result. Requires an even [p].
Float64List matmulF64x2(Float64List a, Float64List b, int n, int m, int p) {
  _checkShapes(a.length, b.length, n, m, p);
  if (p.isOdd) {
    throw ArgumentError('matmulF64x2 needs an even column count.');
  }
  final o = Float64List(n * p);
  final b2 = Float64x2List.view(b.buffer, b.offsetInBytes, b.length ~/ 2);
  final o2 = Float64x2List.view(o.buffer, 0, o.length ~/ 2);
  final p2 = p ~/ 2;
  for (var i = 0; i < n; i++) {
    final aRow = i * m;
    final oRow = i * p2;
    for (var k = 0; k < m; k++) {
      final aik = Float64x2.splat(a[aRow + k]);
      final bRow = k * p2;
      for (var j = 0; j < p2; j++) {
        o2[oRow + j] += b2[bRow + j] * aik;
      }
    }
  }
  return o;
}
