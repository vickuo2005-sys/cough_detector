package com.example.sound_detector_app

import android.media.*
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlin.concurrent.thread
import kotlin.math.sqrt

class MainActivity: FlutterActivity() {

    private val CHANNEL = "sound_channel"
    private var isRecording = false
    private var threshold = 300.0
    private var methodChannel: MethodChannel? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        methodChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)

        methodChannel?.setMethodCallHandler { call, result ->
            when (call.method) {
                "startListening" -> {
                    threshold = call.argument<Double>("threshold") ?: 300.0
                    startAudio()
                    result.success(null)
                }
                "stopListening" -> {
                    isRecording = false
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun startAudio() {
        isRecording = true

        val sampleRate = 16000
        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        val buffer = ShortArray(bufferSize)

        audioRecord.startRecording()

        thread {
            while (isRecording) {
                val read = audioRecord.read(buffer, 0, buffer.size)

                var sum = 0.0
                for (i in 0 until read) {
                    sum += buffer[i] * buffer[i]
                }

                val rms = sqrt(sum / read)

                if (rms > threshold) {
                    methodChannel?.invokeMethod("sound_detected", null)
                } else {
                    methodChannel?.invokeMethod("no_sound", null)
                }

                Thread.sleep(200)
            }

            audioRecord.stop()
            audioRecord.release()
        }
    }
}
