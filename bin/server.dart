import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:dart_ml_lab/matrix.dart';
import 'package:dart_ml_lab/mlp.dart';
import 'package:shelf/shelf.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;
import 'package:shelf_router/shelf_router.dart';

const modelPath = 'model.json';
const defaultPort = 8080;
const maxRows = 10000;
const jsonHeaders = {'content-type': 'application/json'};

Response jsonResponse(int status, Object body) =>
    Response(status, body: jsonEncode(body), headers: jsonHeaders);

Response badRequest(String message) => jsonResponse(400, {'error': message});

/// Thrown by [parseInputs] with a message that is safe to show the client.
class ValidationError implements Exception {
  final String message;
  ValidationError(this.message);
}

/// Validates a `/predict` body and converts it to an input matrix.
Matrix parseInputs(String body, int features) {
  final Object? decoded;
  try {
    decoded = jsonDecode(body);
  } on FormatException {
    throw ValidationError('Request body is not valid JSON.');
  }
  if (decoded is! Map<String, dynamic> || !decoded.containsKey('inputs')) {
    throw ValidationError('Missing "inputs" field.');
  }
  final inputs = decoded['inputs'];
  if (inputs is! List) {
    throw ValidationError('"inputs" must be a list of rows.');
  }
  if (inputs.isEmpty) {
    throw ValidationError('"inputs" must not be empty.');
  }
  if (inputs.length > maxRows) {
    throw ValidationError(
      '"inputs" has ${inputs.length} rows; the maximum is $maxRows.',
    );
  }
  final x = Matrix(inputs.length, features);
  for (var r = 0; r < inputs.length; r++) {
    final row = inputs[r];
    if (row is! List || row.length != features) {
      throw ValidationError('inputs[$r] must be a list of $features numbers.');
    }
    for (var c = 0; c < features; c++) {
      final v = row[c];
      // jsonDecode turns an out-of-range literal such as 1e999 into infinity.
      if (v is! num || !v.isFinite) {
        throw ValidationError('inputs[$r][$c] is not a finite number.');
      }
      x.data[r * features + c] = v.toDouble();
    }
  }
  return x;
}

Handler buildHandler(MLP model, int isolateId) {
  final features = model.sizes.first;
  final classes = model.sizes.last;

  final router = Router(
    notFoundHandler: (_) => jsonResponse(404, {'error': 'Not found.'}),
  );

  router.get('/health', (Request request) {
    return jsonResponse(200, {
      'status': 'ok',
      'parameters': model.parameterCount,
      'isolate': isolateId,
    });
  });

  router.post('/predict', (Request request) async {
    final Matrix x;
    try {
      x = parseInputs(await request.readAsString(), features);
    } on ValidationError catch (e) {
      return badRequest(e.message);
    }

    final watch = Stopwatch()..start();
    final probs = model.predictProba(x);
    watch.stop();

    final predicted = argmaxRows(probs);
    return jsonResponse(200, {
      'predictions': [
        for (var r = 0; r < x.rows; r++)
          {
            'class': predicted[r],
            'probabilities': probs.data.sublist(r * classes, (r + 1) * classes),
          },
      ],
      'count': x.rows,
      'inference_us': watch.elapsedMicroseconds,
    });
  });

  // Anything unexpected is logged server-side and reported generically, so
  // stack traces and internal messages never reach the client.
  Handler catchErrors(Handler inner) => (request) async {
        try {
          return await inner(request);
        } catch (error, stack) {
          stderr
              .writeln('Unhandled error in isolate $isolateId: $error\n$stack');
          return jsonResponse(500, {'error': 'Internal server error.'});
        }
      };

  return const Pipeline()
      .addMiddleware(logRequests())
      .addMiddleware(catchErrors)
      .addHandler(router.call);
}

/// Loads the model and serves it. Runs once per isolate; with `shared: true`
/// the OS distributes incoming connections across all listening isolates.
Future<void> serve(int port, int isolateId, {required bool shared}) async {
  final json = jsonDecode(File(modelPath).readAsStringSync());
  final model = MLP.fromJson(json as Map<String, dynamic>);
  final server = await shelf_io.serve(
    buildHandler(model, isolateId),
    InternetAddress.loopbackIPv4,
    port,
    shared: shared,
  );
  print('isolate $isolateId listening on '
      'http://${server.address.host}:${server.port}');
}

/// Reads `--name value` or `--name=value`; exits with code 64 on bad input.
int intFlag(List<String> args, String name, int fallback) {
  String? raw;
  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (arg == '--$name' && i + 1 < args.length) {
      raw = args[i + 1];
    } else if (arg.startsWith('--$name=')) {
      raw = arg.substring(name.length + 3);
    }
  }
  if (raw == null) return fallback;
  final value = int.tryParse(raw);
  if (value == null || value < 1) {
    stderr.writeln('--$name must be a positive integer, got "$raw".');
    exit(64);
  }
  return value;
}

Future<void> main(List<String> args) async {
  final port = intFlag(args, 'port', defaultPort);
  final isolates = intFlag(args, 'isolates', 1);

  if (!File(modelPath).existsSync()) {
    stderr.writeln('$modelPath not found: run `dart run bin/train.dart` first');
    exit(1);
  }

  if (isolates == 1) {
    await serve(port, 0, shared: false);
    return;
  }
  for (var id = 0; id < isolates; id++) {
    await Isolate.spawn(
      (int id) => serve(port, id, shared: true),
      id,
      errorsAreFatal: true,
    );
  }
  // The spawned isolates own the listening sockets; keep the process alive.
  ReceivePort();
}
