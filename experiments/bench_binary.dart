// Experiment 4: binary serialization vs JSON, on size and speed.
//
// Measures the trained model.json (4,547 parameters) and a larger untrained
// network, to show how each format scales. Timings are in-memory (encode to
// bytes, decode from bytes), so they exclude disk I/O for both formats.
//
//   dart run experiments/bench_binary.dart
//   dart compile exe experiments/bench_binary.dart -o build/bench_binary

import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';

import 'binary_format.dart';
import 'timing.dart';

const modelPath = 'model.json';
const largeSizes = [784, 512, 512, 10];
const largeSeed = 42;
const inputSeed = 5;

var sink = 0;

String kb(int bytes) => '${(bytes / 1024).toStringAsFixed(1)} KB'.padLeft(12);
String ms(double v) => v.toStringAsFixed(4).padLeft(11);

double maxDifference(MLP a, MLP b) {
  final x = Matrix.uniform(64, a.sizes.first, Random(inputSeed), 1);
  final p = a.predictProba(x).data, q = b.predictProba(x).data;
  var worst = 0.0;
  for (var i = 0; i < p.length; i++) {
    worst = max(worst, (p[i] - q[i]).abs());
  }
  return worst;
}

void report(String name, MLP model) {
  print('\n$name: ${model.sizes}, ${model.parameterCount} parameters');
  print('${'format'.padRight(16)}${'size'.padLeft(12)}'
      '${'bytes/param'.padLeft(13)}${'encode ms'.padLeft(11)}'
      '${'decode ms'.padLeft(11)}  max |dp| vs original');

  void row(
    String format,
    Uint8List Function() encode,
    MLP Function(Uint8List) decode,
  ) {
    final bytes = encode();
    final encodeMs = measureMs(() => sink += encode().length);
    final decodeMs = measureMs(() => sink += decode(bytes).parameterCount);
    print('${format.padRight(16)}${kb(bytes.length)}'
        '${(bytes.length / model.parameterCount).toStringAsFixed(2).padLeft(13)}'
        '${ms(encodeMs)}${ms(decodeMs)}  '
        '${maxDifference(model, decode(bytes)).toStringAsExponential(2)}');
  }

  row(
    'JSON',
    () => utf8.encode(jsonEncode(model.toJson())),
    (b) => MLP.fromJson(jsonDecode(utf8.decode(b)) as Map<String, dynamic>),
  );
  row('binary float64', () => encodeModel(model), decodeModel);
  row('binary float32', () => encodeModel(model, float32: true), decodeModel);
}

void main() {
  if (!File(modelPath).existsSync()) {
    stderr.writeln('$modelPath not found: run `dart run bin/train.dart` first');
    exit(1);
  }
  print('serialization (Dart ${Platform.version.split(' ').first})');
  final trained = MLP.fromJson(
    jsonDecode(File(modelPath).readAsStringSync()) as Map<String, dynamic>,
  );
  report('trained model', trained);
  report('large model', MLP(largeSizes, Random(largeSeed)));
  print('\nsink: $sink');
}
