import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:dart_ml_lab/dataset.dart';
import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';

// Data.
const pointsPerClass = 300;
const classCount = 3;
const noise = 0.2;
const datasetSeed = 7;
const trainFraction = 0.8;

// Model.
const layerSizes = [2, 64, 64, 3];
const modelSeed = 42;

// Training.
const batchSize = 32;
const learningRate = 0.2;
const epochs = 400;
const shuffleSeed = 1;
const logEvery = 50;

// Inference benchmark.
const warmupCalls = 1000;
const singleCalls = 100000;
const batchRows = 10000;
const benchSeed = 99;

const modelPath = 'model.json';

double accuracy(MLP model, Matrix x, List<int> y) {
  final predicted = model.predictClasses(x);
  var correct = 0;
  for (var i = 0; i < y.length; i++) {
    if (predicted[i] == y[i]) correct++;
  }
  return correct / y.length;
}

String pct(double v) => '${(v * 100).toStringAsFixed(2)}%';

void main() {
  // Dataset and split share one generator so the whole data pipeline is
  // reproducible from a single seed.
  final dataRng = Random(datasetSeed);
  final data = makeSpiral(
    pointsPerClass: pointsPerClass,
    classes: classCount,
    noise: noise,
    rng: dataRng,
  );
  final order = List<int>.generate(data.y.length, (i) => i)..shuffle(dataRng);
  final trainCount = (order.length * trainFraction).round();
  final trainIdx = order.sublist(0, trainCount);
  final testIdx = order.sublist(trainCount);
  final xTrain = data.x.rowSlice(trainIdx);
  final yTrain = [for (final i in trainIdx) data.y[i]];
  final xTest = data.x.rowSlice(testIdx);
  final yTest = [for (final i in testIdx) data.y[i]];

  final model = MLP(layerSizes, Random(modelSeed));
  print('dart_ml_lab training (Dart ${Platform.version.split(' ').first})');
  print('data:  ${data.y.length} points, $trainCount train / '
      '${testIdx.length} test, $classCount classes, noise $noise');
  print('model: $layerSizes, ${model.parameterCount} parameters');
  print('sgd:   batch $batchSize, lr $learningRate, $epochs epochs\n');

  final shuffleRng = Random(shuffleSeed);
  final batchOrder = List<int>.generate(trainCount, (i) => i);
  var finalLoss = 0.0;
  final trainWatch = Stopwatch()..start();
  for (var epoch = 1; epoch <= epochs; epoch++) {
    batchOrder.shuffle(shuffleRng);
    var lossSum = 0.0;
    for (var start = 0; start < trainCount; start += batchSize) {
      final end = min(start + batchSize, trainCount);
      final idx = batchOrder.sublist(start, end);
      final loss = model.trainBatch(
        xTrain.rowSlice(idx),
        [for (final i in idx) yTrain[i]],
        learningRate,
      );
      // Weight by batch size so the short final batch doesn't skew the mean.
      lossSum += loss * idx.length;
    }
    finalLoss = lossSum / trainCount;
    if (epoch == 1 || epoch % logEvery == 0) {
      print('epoch ${epoch.toString().padLeft(3)}  '
          'loss ${finalLoss.toStringAsFixed(6)}  '
          'train ${pct(accuracy(model, xTrain, yTrain))}  '
          'test ${pct(accuracy(model, xTest, yTest))}');
    }
  }
  trainWatch.stop();
  final trainMs = trainWatch.elapsedMicroseconds / 1000;
  final trainAcc = accuracy(model, xTrain, yTrain);
  final testAcc = accuracy(model, xTest, yTest);

  // --- Inference benchmark -------------------------------------------------
  // Inputs are built up front so only the model call is timed. Results are
  // folded into `sink` and printed, so the compiler can't discard the work.
  final benchRng = Random(benchSeed);
  final singles = [
    for (var i = 0; i < 256; i++)
      Matrix.uniform(1, layerSizes.first, benchRng, 1),
  ];
  var sink = 0.0;
  for (var i = 0; i < warmupCalls; i++) {
    sink += model.predictProba(singles[i & 255]).data[0];
  }
  final singleWatch = Stopwatch()..start();
  for (var i = 0; i < singleCalls; i++) {
    sink += model.predictProba(singles[i & 255]).data[0];
  }
  singleWatch.stop();
  final singleUs = singleWatch.elapsedMicroseconds / singleCalls;

  final bigBatch = Matrix.uniform(batchRows, layerSizes.first, benchRng, 1);
  final batchWatch = Stopwatch()..start();
  final batchProbs = model.predictProba(bigBatch);
  batchWatch.stop();
  for (var i = 0; i < batchRows; i++) {
    sink += batchProbs.data[i * classCount];
  }
  final batchMs = batchWatch.elapsedMicroseconds / 1000;
  final batchUsPerRow = batchWatch.elapsedMicroseconds / batchRows;

  final encoded = jsonEncode(model.toJson());
  File(modelPath).writeAsStringSync(encoded);
  final modelKb = File(modelPath).lengthSync() / 1024;

  print('\n=== summary ===');
  print('final loss:            ${finalLoss.toStringAsFixed(6)}');
  print('train accuracy:        ${pct(trainAcc)}');
  print('test accuracy:         ${pct(testAcc)}');
  print('training time:         ${trainMs.toStringAsFixed(1)} ms '
      '(${(trainMs / epochs).toStringAsFixed(3)} ms/epoch)');
  print('single prediction:     ${singleUs.toStringAsFixed(3)} us mean '
      'over $singleCalls calls');
  print('batch of $batchRows:       ${batchMs.toStringAsFixed(3)} ms total, '
      '${batchUsPerRow.toStringAsFixed(3)} us/row');
  print('benchmark sink:        ${sink.toStringAsFixed(6)}');
  print('parameters:            ${model.parameterCount}');
  print('$modelPath:            ${modelKb.toStringAsFixed(1)} KB');
}
