import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

const inputSeed = 5;

/// Reads `--name value` or `--name=value`.
String flag(List<String> args, String name, String fallback) {
  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (arg == '--$name' && i + 1 < args.length) return args[i + 1];
    if (arg.startsWith('--$name=')) return arg.substring(name.length + 3);
  }
  return fallback;
}

int intFlag(List<String> args, String name, int fallback) {
  final raw = flag(args, name, '$fallback');
  final value = int.tryParse(raw);
  if (value == null || value < 1) {
    stderr.writeln('--$name must be a positive integer, got "$raw".');
    exit(64);
  }
  return value;
}

/// Returns the [p]-th percentile (nearest rank) of an ascending list.
int percentile(List<int> sorted, double p) =>
    sorted[max(0, (p / 100 * sorted.length).ceil() - 1)];

String ms(int micros) => (micros / 1000).toStringAsFixed(3);

Future<void> main(List<String> args) async {
  final url = Uri.parse(flag(args, 'url', 'http://127.0.0.1:8080/predict'));
  final concurrency = intFlag(args, 'concurrency', 50);
  final requests = intFlag(args, 'requests', 5000);
  final batch = intFlag(args, 'batch', 1);

  // One fixed body, reused for every request, so the generator spends its
  // time on I/O rather than on encoding JSON.
  final rng = Random(inputSeed);
  final body = utf8.encode(jsonEncode({
    'inputs': [
      for (var i = 0; i < batch; i++)
        [rng.nextDouble() * 2 - 1, rng.nextDouble() * 2 - 1],
    ],
  }));

  final client = HttpClient()..maxConnectionsPerHost = concurrency;
  final latencies = <int>[];
  var errors = 0;
  var next = 0;

  Future<bool> send() async {
    try {
      final request = await client.postUrl(url);
      request.headers.contentType = ContentType.json;
      request.contentLength = body.length;
      request.add(body);
      final response = await request.close();
      await response.drain<void>();
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // Warm up connections and the server's code paths; not measured.
  await Future.wait([for (var i = 0; i < concurrency; i++) send()]);

  Future<void> worker() async {
    while (next < requests) {
      next++;
      final watch = Stopwatch()..start();
      final ok = await send();
      watch.stop();
      if (ok) {
        latencies.add(watch.elapsedMicroseconds);
      } else {
        errors++;
      }
    }
  }

  final total = Stopwatch()..start();
  await Future.wait([for (var i = 0; i < concurrency; i++) worker()]);
  total.stop();
  client.close();

  final seconds = total.elapsedMicroseconds / 1e6;
  print('url:          $url');
  print('requests:     $requests  (concurrency $concurrency, '
      'batch $batch, body ${body.length} bytes)');
  print('duration:     ${seconds.toStringAsFixed(3)} s');
  print('throughput:   ${(requests / seconds).toStringAsFixed(0)} req/s  '
      '(${(requests * batch / seconds).toStringAsFixed(0)} rows/s)');
  if (latencies.isNotEmpty) {
    latencies.sort();
    print('latency ms:   p50 ${ms(percentile(latencies, 50))}  '
        'p95 ${ms(percentile(latencies, 95))}  '
        'p99 ${ms(percentile(latencies, 99))}  '
        'max ${ms(latencies.last)}');
  }
  print('errors:       $errors');
  if (errors > 0) exitCode = 1;
}
