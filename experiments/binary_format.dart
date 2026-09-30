/// A compact binary encoding of an [MLP], as an alternative to `model.json`.
///
/// Layout (all little-endian):
///
///     offset  size  field
///     0       4     magic "DMLP"
///     4       4     format version (uint32) = 1
///     8       4     element width in bytes (uint32): 8 = float64, 4 = float32
///     12      4     number of entries in `sizes` (uint32)
///     16      4*k   sizes (uint32 each)
///     ...           zero padding to a multiple of 8 bytes
///     ...           for each layer: weights (row-major), then biases
///
/// Float64 is lossless. Float32 (an extra, not on the original list) halves
/// the file and rounds every parameter.
library;

import 'dart:typed_data';

import 'package:dart_ml_lab/mlp.dart';

const _magic = 0x504C4D44; // "DMLP" read as a little-endian uint32.
const _version = 1;

int _headerLength(int sizeCount) => (16 + 4 * sizeCount + 7) & ~7;

Uint8List encodeModel(MLP model, {bool float32 = false}) {
  final width = float32 ? 4 : 8;
  final header = _headerLength(model.sizes.length);
  final bytes = Uint8List(header + model.parameterCount * width);
  final view = ByteData.sublistView(bytes);
  view.setUint32(0, _magic, Endian.little);
  view.setUint32(4, _version, Endian.little);
  view.setUint32(8, width, Endian.little);
  view.setUint32(12, model.sizes.length, Endian.little);
  for (var i = 0; i < model.sizes.length; i++) {
    view.setUint32(16 + 4 * i, model.sizes[i], Endian.little);
  }
  var offset = header;
  for (var l = 0; l < model.layerCount; l++) {
    for (final m in [model.weights[l], model.biases[l]]) {
      final d = m.data;
      if (!float32 && Endian.host == Endian.little) {
        // The in-memory representation already is the file format.
        bytes.setRange(
          offset,
          offset + d.lengthInBytes,
          d.buffer.asUint8List(d.offsetInBytes, d.lengthInBytes),
        );
        offset += d.lengthInBytes;
      } else {
        for (var i = 0; i < d.length; i++, offset += width) {
          if (float32) {
            view.setFloat32(offset, d[i], Endian.little);
          } else {
            view.setFloat64(offset, d[i], Endian.little);
          }
        }
      }
    }
  }
  return bytes;
}

/// Decodes [encodeModel] output, validating the header and the total length.
///
/// The model is built through `MLP.fromJson`, the only public way to construct
/// one from existing parameters. That costs an extra copy and a per-element
/// check, which a constructor inside `lib/` would avoid.
MLP decodeModel(Uint8List bytes) {
  if (bytes.length < 16) {
    throw const FormatException('Model file is too short to hold a header.');
  }
  final view = ByteData.sublistView(bytes);
  if (view.getUint32(0, Endian.little) != _magic) {
    throw const FormatException('Not a DMLP model file (bad magic).');
  }
  final version = view.getUint32(4, Endian.little);
  if (version != _version) {
    throw FormatException('Unsupported binary format version: $version.');
  }
  final width = view.getUint32(8, Endian.little);
  if (width != 4 && width != 8) {
    throw FormatException('Unsupported element width: $width.');
  }
  final sizeCount = view.getUint32(12, Endian.little);
  final header = _headerLength(sizeCount);
  if (sizeCount < 2 || bytes.length < header) {
    throw const FormatException('Model header is truncated or invalid.');
  }
  final sizes = [
    for (var i = 0; i < sizeCount; i++)
      view.getUint32(16 + 4 * i, Endian.little),
  ];
  var parameters = 0;
  for (var l = 0; l < sizeCount - 1; l++) {
    parameters += (sizes[l] + 1) * sizes[l + 1];
  }
  if (bytes.length != header + parameters * width) {
    throw FormatException(
      'Expected ${header + parameters * width} bytes for sizes $sizes, '
      'got ${bytes.length}.',
    );
  }

  var offset = header;
  List<double> read(int count) {
    final List<double> values;
    if (Endian.host == Endian.little && bytes.offsetInBytes % width == 0) {
      // A typed view straight onto the file bytes: no per-element decoding.
      final start = bytes.offsetInBytes + offset;
      values = width == 8
          ? bytes.buffer.asFloat64List(start, count)
          : bytes.buffer.asFloat32List(start, count);
    } else {
      values = [
        for (var i = 0; i < count; i++)
          width == 8
              ? view.getFloat64(offset + i * width, Endian.little)
              : view.getFloat32(offset + i * width, Endian.little),
      ];
    }
    offset += count * width;
    return values;
  }

  final weights = <List<double>>[];
  final biases = <List<double>>[];
  for (var l = 0; l < sizeCount - 1; l++) {
    weights.add(read(sizes[l] * sizes[l + 1]));
    biases.add(read(sizes[l + 1]));
  }
  return MLP.fromJson({
    'format_version': MLP.formatVersion,
    'sizes': sizes,
    'weights': weights,
    'biases': biases,
  });
}
