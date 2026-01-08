import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(home: TensorProbePage()));
}

class TensorProbePage extends StatefulWidget {
  const TensorProbePage({super.key});

  @override
  State<TensorProbePage> createState() => _TensorProbePageState();
}

class _TensorProbePageState extends State<TensorProbePage> {
  String status = "Loading...";
  Interpreter? interpreter;

  @override
  void initState() {
    super.initState();
    _loadAndPrint();
  }

 Future<void> _loadAndPrint() async {
  try {
    // 1) 先確認 asset 真的讀得到
    final bytes = await rootBundle.load('assets/models/yamnet.tflite');
    debugPrint("Asset bytes length = ${bytes.lengthInBytes}");

    // 2) 直接用 bytes 建立 Interpreter（最穩，避開 fromAsset 路徑問題）
    final modelBytes = bytes.buffer.asUint8List(
      bytes.offsetInBytes,
      bytes.lengthInBytes,
    );
    interpreter = Interpreter.fromBuffer(modelBytes);

    final inCount = interpreter!.getInputTensors().length;
    final outCount = interpreter!.getOutputTensors().length;

    debugPrint("=== YAMNet TFLite Tensor Info ===");
    debugPrint("Inputs: $inCount");
    for (int i = 0; i < inCount; i++) {
      final t = interpreter!.getInputTensor(i);
      debugPrint("  [IN $i] name=${t.name} type=${t.type} shape=${t.shape}");
    }

    debugPrint("Outputs: $outCount");
    int? embIndex;
    for (int i = 0; i < outCount; i++) {
      final t = interpreter!.getOutputTensor(i);
      debugPrint("  [OUT $i] name=${t.name} type=${t.type} shape=${t.shape}");

      final shape = t.shape;
      if (shape.isNotEmpty && shape.last == 1024) {
        embIndex ??= i;
      }
    }

    debugPrint("Embeddings candidate output index: ${embIndex ?? 'NOT FOUND'}");
    debugPrint("================================");

    setState(() {
      status =
          "Loaded.\nAsset bytes=${bytes.lengthInBytes}\nEmbeddings index: ${embIndex ?? 'NOT FOUND'}";
    });
  } catch (e) {
    setState(() {
      status = "Failed: $e";
    });
  }
}
  @override
  void dispose() {
    interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("YAMNet Tensor Probe")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(status, style: const TextStyle(fontSize: 16)),
      ),
    );
  }
}
