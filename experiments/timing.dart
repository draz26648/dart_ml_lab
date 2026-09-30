const _warmupMicros = 100 * 1000;
const _measureMicros = 200 * 1000;

/// Returns the mean milliseconds per call of [op].
///
/// Warms up for ~100 ms, then repeats for at least ~200 ms; each phase runs
/// [op] at least once, so a multi-second operation is executed twice in total.
/// [op] must feed its result into a printed sink so the work can't be removed.
double measureMs(void Function() op) {
  final warmup = Stopwatch()..start();
  do {
    op();
  } while (warmup.elapsedMicroseconds < _warmupMicros);

  final watch = Stopwatch()..start();
  var reps = 0;
  do {
    op();
    reps++;
  } while (watch.elapsedMicroseconds < _measureMicros);
  return watch.elapsedMicroseconds / 1000 / reps;
}

double gflops(int n, double ms) => 2.0 * n * n * n / (ms / 1000) / 1e9;
