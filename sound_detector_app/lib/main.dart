import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:vibration/vibration.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: SoundPage());
  }
}

class SoundPage extends StatefulWidget {
  @override
  _SoundPageState createState() => _SoundPageState();
}

class _SoundPageState extends State<SoundPage> {
  static const platform = MethodChannel('sound_channel');

  String status = "未啟動";
  bool isListening = false;
  double threshold = 300;

  Future<void> startListening() async {
    await platform.invokeMethod('startListening', {"threshold": threshold});
    setState(() {
      isListening = true;
      status = "監聽中...";
    });
    listenCallback();
  }

  Future<void> stopListening() async {
    await platform.invokeMethod('stopListening');
    setState(() {
      isListening = false;
      status = "已停止";
    });
  }

  void listenCallback() {
    platform.setMethodCallHandler((call) async {
      if (call.method == "sound_detected") {
        setState(() => status = "⚠️ 有聲音！");
        if (await Vibration.hasVibrator() ?? false) {
          Vibration.vibrate(duration: 200);
        }
      } else {
        setState(() => status = "安靜中");
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("聲音偵測APP")),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Center(child: Text(status, style: TextStyle(fontSize: 24))),
          Slider(
            value: threshold,
            min: 50,
            max: 2000,
            divisions: 50,
            label: threshold.toInt().toString(),
            onChanged: (value) {
              setState(() => threshold = value);
            },
          ),
          ElevatedButton(
            onPressed: isListening ? stopListening : startListening,
            child: Text(isListening ? "停止" : "開始偵測"),
          )
        ],
      ),
    );
  }
}
