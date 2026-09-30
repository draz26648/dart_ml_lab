// Experiment 3: Adam vs SGD, measured in epochs and in wall time.
//
// Same data, split, architecture, init seed, batch size and shuffle seed as
// bin/train.dart; only the optimizer differs. Accuracy is evaluated after
// every epoch with the training clock stopped, so the reported times cover
// optimisation only.
//
//   dart run experiments/bench_adam.dart
//   dart compile exe experiments/bench_adam.dart -o build/bench_adam

import 'dart:io';
import 'dart:math';

import 'package:dart_ml_lab/dataset.dart';
import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';

import 'adam.dart';

// Identical to bin/train.dart.
const pointsPerClass = 300;
const classCount = 3;
const noise = 0.2;
const datasetSeed = 7;
const trainFraction = 0.8;
const layerSizes = [2, 64, 64, 3];
const modelSeed = 42;
const batchSize = 32;
const epochs = 400;
const shuffleSeed = 1;
const sgdLearningRate = 0.2;

// Adam step sizes to try. 0.001 is the paper's default; the rest bracket it.
const adamLearningRates = [0.001, 0.003, 0.01, 0.03];

// "Same accuracy" targets, checked at the end of each epoch.
const lossTargets = [0.05, 0.02];
const testAccuracyTarget = 0.97;

typedef Step = double Function(MLP model, Matrix x, List<int> labels);

double accuracy(MLP model, Matrix x, List<int> y) {
  final predicted = model.predictClasses(x);
  var correct = 0;
  for (var i = 0; i < y.length; i++) {
    if (predicted[i] == y[i]) correct++;
  }
  return correct / y.length;
}

String cell(int? epoch, double? ms) => epoch == null
    ? 'never'.padLeft(18)
    : '$epoch ep / ${ms!.toStringAsFixed(0)} ms'.padLeft(18);

void main() {
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

  print('Adam vs SGD (Dart ${Platform.version.split(' ').first}): '
      '$layerSizes, batch $batchSize, $epochs epochs, '
      '$trainCount train / ${testIdx.length} test');
  print('${'optimizer'.padRight(16)}'
      '${[for (final t in lossTargets) 'loss <= $t'.padLeft(18)].join()}'
      '${'test >= ${(testAccuracyTarget * 100).round()}%'.padLeft(18)}'
      '${'final loss'.padLeft(12)}${'final test'.padLeft(12)}'
      '${'best test'.padLeft(11)}${'total ms'.padLeft(10)}'
      '${'ms/epoch'.padLeft(10)}');

  void run(String name, Step Function(MLP model) makeStep) {
    final model = MLP(layerSizes, Random(modelSeed));
    final step = makeStep(model);
    final shuffleRng = Random(shuffleSeed);
    final batchOrder = List<int>.generate(trainCount, (i) => i);
    final lossHit = List<(int, double)?>.filled(lossTargets.length, null);
    (int, double)? accuracyHit;
    var finalLoss = 0.0, finalTest = 0.0, bestTest = 0.0;
    final watch = Stopwatch();
    for (var epoch = 1; epoch <= epochs; epoch++) {
      watch.start();
      batchOrder.shuffle(shuffleRng);
      var lossSum = 0.0;
      for (var start = 0; start < trainCount; start += batchSize) {
        final end = min(start + batchSize, trainCount);
        final idx = batchOrder.sublist(start, end);
        lossSum += idx.length *
            step(model, xTrain.rowSlice(idx), [for (final i in idx) yTrain[i]]);
      }
      watch.stop();
      final ms = watch.elapsedMicroseconds / 1000;
      finalLoss = lossSum / trainCount;
      finalTest = accuracy(model, xTest, yTest);
      bestTest = max(bestTest, finalTest);
      for (var t = 0; t < lossTargets.length; t++) {
        if (lossHit[t] == null && finalLoss <= lossTargets[t]) {
          lossHit[t] = (epoch, ms);
        }
      }
      if (accuracyHit == null && finalTest >= testAccuracyTarget) {
        accuracyHit = (epoch, ms);
      }
    }
    final totalMs = watch.elapsedMicroseconds / 1000;
    print('${name.padRight(16)}'
        '${[for (final h in lossHit) cell(h?.$1, h?.$2)].join()}'
        '${cell(accuracyHit?.$1, accuracyHit?.$2)}'
        '${finalLoss.toStringAsFixed(6).padLeft(12)}'
        '${'${(finalTest * 100).toStringAsFixed(2)}%'.padLeft(12)}'
        '${'${(bestTest * 100).toStringAsFixed(2)}%'.padLeft(11)}'
        '${totalMs.toStringAsFixed(0).padLeft(10)}'
        '${(totalMs / epochs).toStringAsFixed(2).padLeft(10)}');
  }

  run('SGD lr=$sgdLearningRate',
      (_) => (model, x, y) => model.trainBatch(x, y, sgdLearningRate));
  for (final lr in adamLearningRates) {
    run('Adam lr=$lr', (model) {
      final adam = Adam(model, learningRate: lr);
      return adam.trainBatch;
    });
  }
}
