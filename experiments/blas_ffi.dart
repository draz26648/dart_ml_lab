/// Experiment 6: matmul backed by Apple's Accelerate BLAS through `dart:ffi`.
///
/// THIS FILE DELIBERATELY BREAKS THE PROJECT'S "PURE DART, NO FFI" RULE. It
/// exists only as a contrast, to show how much of the Dart-vs-NumPy gap is the
/// library rather than the language. Nothing in `lib/` or `bin/` imports it.
///
/// It uses only `dart:ffi` (no `package:ffi`), so native memory comes from the
/// C library's own `malloc`/`free`. macOS only.
library;

import 'dart:ffi';
import 'dart:typed_data';

import 'package:dart_ml_lab/matrix.dart';

const _acceleratePath =
    '/System/Library/Frameworks/Accelerate.framework/Accelerate';
const _cblasRowMajor = 101;
const _cblasNoTrans = 111;

typedef _DgemmC = Void Function(
  Int32 order,
  Int32 transA,
  Int32 transB,
  Int32 m,
  Int32 n,
  Int32 k,
  Double alpha,
  Pointer<Double> a,
  Int32 lda,
  Pointer<Double> b,
  Int32 ldb,
  Double beta,
  Pointer<Double> c,
  Int32 ldc,
);
typedef _DgemmDart = void Function(
  int order,
  int transA,
  int transB,
  int m,
  int n,
  int k,
  double alpha,
  Pointer<Double> a,
  int lda,
  Pointer<Double> b,
  int ldb,
  double beta,
  Pointer<Double> c,
  int ldc,
);

/// A double array in native memory, visible to Dart as a [Float64List].
class NativeDoubles {
  final Pointer<Double> pointer;
  final Float64List list;
  NativeDoubles._(this.pointer, this.list);
}

class Blas {
  final _DgemmDart _dgemm;
  final Pointer<Void> Function(int) _malloc;
  final void Function(Pointer<Void>) _free;

  Blas._(this._dgemm, this._malloc, this._free);

  factory Blas.open() {
    final accelerate = DynamicLibrary.open(_acceleratePath);
    final libc = DynamicLibrary.process();
    return Blas._(
      // isLeaf: the call never re-enters Dart, so the VM can skip the
      // safepoint transition and keep the call overhead minimal.
      accelerate.lookupFunction<_DgemmC, _DgemmDart>(
        'cblas_dgemm',
        isLeaf: true,
      ),
      libc.lookupFunction<Pointer<Void> Function(IntPtr),
          Pointer<Void> Function(int)>('malloc'),
      libc.lookupFunction<Void Function(Pointer<Void>),
          void Function(Pointer<Void>)>('free'),
    );
  }

  NativeDoubles allocate(int length) {
    final pointer = _malloc(length * sizeOf<Double>()).cast<Double>();
    if (pointer == nullptr) throw StateError('malloc failed');
    return NativeDoubles._(pointer, pointer.asTypedList(length));
  }

  void free(NativeDoubles block) => _free(block.pointer.cast());

  /// `c = a * b` on buffers that already live in native memory: no copying.
  void dgemm(
    NativeDoubles a,
    NativeDoubles b,
    NativeDoubles c,
    int n,
    int m,
    int p,
  ) {
    _dgemm(_cblasRowMajor, _cblasNoTrans, _cblasNoTrans, n, p, m, 1.0,
        a.pointer, m, b.pointer, p, 0.0, c.pointer, p);
  }

  /// The same API as `Matrix.matmul`: Dart-heap matrices in, a Dart-heap
  /// matrix out. Pays for two copies in, one copy out and three allocations.
  Matrix matmul(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
      throw ArgumentError(
        'Cannot multiply ${a.rows}x${a.cols} by ${b.rows}x${b.cols}.',
      );
    }
    final na = allocate(a.data.length);
    final nb = allocate(b.data.length);
    final nc = allocate(a.rows * b.cols);
    try {
      na.list.setAll(0, a.data);
      nb.list.setAll(0, b.data);
      dgemm(na, nb, nc, a.rows, a.cols, b.cols);
      final out = Matrix(a.rows, b.cols);
      out.data.setAll(0, nc.list);
      return out;
    } finally {
      free(na);
      free(nb);
      free(nc);
    }
  }
}
