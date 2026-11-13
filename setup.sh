#!/usr/bin/env bash
set -e

mkdir -p app/libs
mkdir -p app/src/main/assets/models/whisper-base
mkdir -p app/src/main/java/com/example/myenglishrecorder

curl -fSL -o app/libs/sherpa-onnx-2.0.1.aar \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/sherpa-onnx-2.0.1.aar

curl -fSL -o app/src/main/assets/models/silero_vad.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

curl -fSL -o app/src/main/assets/models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx

curl -fSL -o whisper-base.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2
tar xjf whisper-base.tar.bz2
cp sherpa-onnx-whisper-base/base-encoder.int8.onnx app/src/main/assets/models/whisper-base/
cp sherpa-onnx-whisper-base/base-decoder.int8.onnx app/src/main/assets/models/whisper-base/
cp sherpa-onnx-whisper-base/base-tokens.txt app/src/main/assets/models/whisper-base/
rm -rf whisper-base.tar.bz2 sherpa-onnx-whisper-base

cat << 'EOF' > gradle.properties
org.gradle.jvmargs=-Xmx4096m -XX:MaxMetaspaceSize=1024m -Dfile.encoding=UTF-8
android.useAndroidX=true
android.enableJetifier=true
kotlin.code.style=official
EOF

cat << 'EOF' > settings.gradle.kts
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "MyEnglishRecorder"
include(":app")
EOF

cat << 'EOF' > build.gradle.kts
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}
EOF

cat << 'EOF' > app/build.gradle.kts
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.myenglishrecorder"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.myenglishrecorder"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        noCompress += listOf("onnx", "txt")
    }
}

dependencies {
    implementation(files("libs/sherpa-onnx-2.0.1.aar"))
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
EOF

cat << 'EOF' > app/src/main/AndroidManifest.xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE" />
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
    <uses-permission android:name="android.permission.WAKE_LOCK" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"
        android:maxSdkVersion="28" />

    <application
        android:allowBackup="true"
        android:icon="@android:drawable/ic_btn_speak_now"
        android:label="MyEnglishRecorder"
        android:theme="@style/Theme.AppCompat.Light.DarkActionBar">

        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

        <service
            android:name=".VoiceRecorderService"
            android:enabled="true"
            android:exported="false"
            android:foregroundServiceType="microphone" />
    </application>
</manifest>
EOF

cat << 'EOF' > app/src/main/java/com/example/myenglishrecorder/VoiceAiEngine.kt
package com.example.myenglishrecorder

import android.content.Context
import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.*
import java.io.File

class VoiceAiEngine(private val context: Context) {

    private val assets: AssetManager = context.assets
    private val sampleRate = 16000
    private val speakerName = "owner"
    private val voiceprintFile = File(context.filesDir, "my_voiceprint.bin")

    lateinit var vad: Vad
    private lateinit var speakerExtractor: SpeakerEmbeddingExtractor
    private lateinit var speakerManager: SpeakerEmbeddingManager
    private lateinit var langIdentifier: SpokenLanguageIdentification
    private lateinit var whisperRecognizer: OfflineRecognizer

    var isEnrolled: Boolean = false
        private set

    fun init() {
        vad = Vad(
            assetManager = assets,
            config = VadModelConfig(
                sileroVadModelConfig = SileroVadModelConfig(
                    model = "models/silero_vad.onnx",
                    threshold = 0.5f,
                    minSilenceDuration = 0.5f,
                    minSpeechDuration = 0.35f,
                    windowSize = 512,
                    maxSpeechDuration = 15.0f
                ),
                sampleRate = sampleRate,
                numThreads = 2,
                provider = "cpu"
            )
        )

        speakerExtractor = SpeakerEmbeddingExtractor(
            assetManager = assets,
            config = SpeakerEmbeddingExtractorConfig(
                model = "models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
                numThreads = 2,
                provider = "cpu"
            )
        )
        speakerManager = SpeakerEmbeddingManager(dim = speakerExtractor.dim())
        loadSavedVoiceprint()

        langIdentifier = SpokenLanguageIdentification(
            assetManager = assets,
            config = SpokenLanguageIdentificationConfig(
                whisper = SpokenLanguageIdentificationWhisperConfig(
                    encoder = "models/whisper-base/base-encoder.int8.onnx",
                    decoder = "models/whisper-base/base-decoder.int8.onnx"
                ),
                numThreads = 2,
                provider = "cpu"
            )
        )

        whisperRecognizer = OfflineRecognizer(
            assetManager = assets,
            config = OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = sampleRate, featureDim = 80),
                modelConfig = OfflineModelConfig(
                    whisper = OfflineWhisperModelConfig(
                        encoder = "models/whisper-base/base-encoder.int8.onnx",
                        decoder = "models/whisper-base/base-decoder.int8.onnx",
                        language = "en",
                        task = "transcribe"
                    ),
                    tokens = "models/whisper-base/base-tokens.txt",
                    modelType = "whisper",
                    numThreads = 4,
                    provider = "cpu"
                )
            )
        )
    }

    fun enrollVoice(samples: FloatArray): Boolean {
        val stream = speakerExtractor.createStream()
        stream.acceptWaveform(samples, sampleRate)
        stream.inputFinished()

        if (!speakerExtractor.isReady(stream)) {
            stream.release()
            return false
        }

        val embedding = speakerExtractor.compute(stream)
        stream.release()

        speakerManager.remove(speakerName)
        val ok = speakerManager.add(speakerName, embedding)
        if (ok) {
            saveVoiceprint(embedding)
            isEnrolled = true
        }
        return ok
    }

    fun processSpeechSegment(samples: FloatArray): FilterResult {
        if (!isEnrolled) return FilterResult.Rejected("Voice not registered")
        if (samples.size < (sampleRate * 0.6).toInt()) {
            return FilterResult.Rejected("Too short / noise")
        }

        val spkStream = speakerExtractor.createStream()
        spkStream.acceptWaveform(samples, sampleRate)
        spkStream.inputFinished()
        if (!speakerExtractor.isReady(spkStream)) {
            spkStream.release()
            return FilterResult.Rejected("Unclear voice")
        }
        val embedding = speakerExtractor.compute(spkStream)
        spkStream.release()

        val isOwner = speakerManager.verify(speakerName, embedding, 0.35f)
        if (!isOwner) {
            return FilterResult.Rejected("Other person's voice")
        }

        val langStream = langIdentifier.createStream()
        langStream.acceptWaveform(samples, sampleRate)
        val detectedLang = langIdentifier.compute(langStream)
        langStream.release()

        if (detectedLang != "en") {
            return FilterResult.Rejected("Non-English ($detectedLang)")
        }

        val asrStream = whisperRecognizer.createStream()
        asrStream.acceptWaveform(samples, sampleRate)
        whisperRecognizer.decode(asrStream)
        val text = whisperRecognizer.getResult(asrStream).text.trim()
        asrStream.release()

        return if (text.isNotEmpty()) {
            FilterResult.Accepted(text)
        } else {
            FilterResult.Rejected("Empty speech")
        }
    }

    private fun saveVoiceprint(emb: FloatArray) {
        val bytes = java.nio.ByteBuffer.allocate(emb.size * 4)
        emb.forEach { bytes.putFloat(it) }
        voiceprintFile.writeBytes(bytes.array())
    }

    private fun loadSavedVoiceprint() {
        if (!voiceprintFile.exists()) return
        val bytes = voiceprintFile.readBytes()
        val buf = java.nio.ByteBuffer.wrap(bytes)
        val emb = FloatArray(bytes.size / 4) { buf.getFloat() }
        if (emb.size == speakerExtractor.dim()) {
            speakerManager.add(speakerName, emb)
            isEnrolled = true
        }
    }
}

sealed class FilterResult {
    data class Accepted(val text: String) : FilterResult()
    data class Rejected(val reason: String) : FilterResult()
}
EOF

cat << 'EOF' > app/src/main/java/com/example/myenglishrecorder/AudioStorageHelper.kt
package com.example.myenglishrecorder

import android.content.ContentValues
import android.content.Context
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*

data class SavedFilesResult(
    val audioFileName: String?,
    val textFileName: String?
)

class AudioStorageHelper(private val context: Context) {

    private val lock = Any()
    private var pcmFile: File? = null
    private var pcmOut: BufferedOutputStream? = null
    private var totalAudioBytes: Long = 0

    private var textFile: File? = null
    private val textBuffer = StringBuilder()

    fun startNewSession() = synchronized(lock) {
        val rawAudio = File(context.cacheDir, "current_english_only.pcm")
        if (rawAudio.exists()) rawAudio.delete()
        pcmFile = rawAudio
        pcmOut = BufferedOutputStream(FileOutputStream(rawAudio))
        totalAudioBytes = 0

        val rawText = File(context.cacheDir, "current_english_only.txt")
        if (rawText.exists()) rawText.delete()
        textFile = rawText
        textBuffer.clear()
    }

    fun appendEnglishSamples(samples: FloatArray) = synchronized(lock) {
        val out = pcmOut ?: return
        val byteBuf = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (s in samples) {
            val clamped = s.coerceIn(-1.0f, 1.0f)
            byteBuf.putShort((clamped * 32767).toInt().toShort())
        }
        val bytes = byteBuf.array()
        out.write(bytes)
        out.flush()
        totalAudioBytes += bytes.size
    }

    fun appendEnglishText(sentence: String) = synchronized(lock) {
        if (sentence.isBlank()) return
        if (textBuffer.isNotEmpty()) textBuffer.append(" ")
        textBuffer.append(sentence.trim())
        textFile?.writeText(textBuffer.toString(), Charsets.UTF_8)
    }

    fun exportBothToDownloads(clearAfterSave: Boolean): SavedFilesResult = synchronized(lock) {
        pcmOut?.flush()
        val rawAudio = pcmFile
        val currentBytes = totalAudioBytes
        val currentText = textBuffer.toString().trim()

        if ((rawAudio == null || !rawAudio.exists() || currentBytes == 0L) && currentText.isEmpty()) {
            return SavedFilesResult(null, null)
        }

        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val audioName = "MyEnglish_$timeStamp.wav"
        val textName = "MyEnglish_$timeStamp.txt"

        var savedAudioName: String? = null
        var savedTextName: String? = null

        if (rawAudio != null && rawAudio.exists() && currentBytes > 0L) {
            val header = createWavHeader(currentBytes, 16000, 1, 16)
            val ok = writeBytesToDownloads(audioName, "audio/wav") { out ->
                out.write(header)
                FileInputStream(rawAudio).use { input ->
                    val buf = ByteArray(8192)
                    var remaining = currentBytes
                    while (remaining > 0) {
                        val toRead = minOf(buf.size.toLong(), remaining).toInt()
                        val read = input.read(buf, 0, toRead)
                        if (read <= 0) break
                        out.write(buf, 0, read)
                        remaining -= read
                    }
                }
            }
            if (ok) savedAudioName = audioName
        }

        if (currentText.isNotEmpty()) {
            val ok = writeBytesToDownloads(textName, "text/plain") { out ->
                out.write(currentText.toByteArray(Charsets.UTF_8))
            }
            if (ok) savedTextName = textName
        }

        if (clearAfterSave) {
            pcmOut?.close()
            pcmOut = null
            rawAudio?.delete()
            textFile?.delete()
            totalAudioBytes = 0
            textBuffer.clear()
        }

        return SavedFilesResult(savedAudioName, savedTextName)
    }

    private fun writeBytesToDownloads(
        fileName: String,
        mimeType: String,
        writer: (OutputStream) -> Unit
    ): Boolean {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = context.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(MediaStore.Downloads.MIME_TYPE, mimeType)
                    put(
                        MediaStore.Downloads.RELATIVE_PATH,
                        "${Environment.DIRECTORY_DOWNLOADS}/MyEnglishRecordings"
                    )
                    put(MediaStore.Downloads.IS_PENDING, 1)
                }
                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: return false
                resolver.openOutputStream(uri)?.use { out -> writer(out) } ?: return false

                values.clear()
                values.put(MediaStore.Downloads.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
                true
            } else {
                @Suppress("DEPRECATION")
                val dir = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                    "MyEnglishRecordings"
                )
                if (!dir.exists()) dir.mkdirs()
                val outFile = File(dir, fileName)
                FileOutputStream(outFile).use { out -> writer(out) }
                MediaScannerConnection.scanFile(context, arrayOf(outFile.absolutePath), null, null)
                true
            }
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    private fun createWavHeader(pcmBytes: Long, sampleRate: Int, channels: Int, bitsPerSample: Int): ByteArray {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val totalDataLen = pcmBytes + 36
        return ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray())
            putInt(totalDataLen.toInt())
            put("WAVE".toByteArray())
            put("fmt ".toByteArray())
            putInt(16)
            putShort(1.toShort())
            putShort(channels.toShort())
            putInt(sampleRate)
            putInt(byteRate)
            putShort((channels * bitsPerSample / 8).toShort())
            putShort(bitsPerSample.toShort())
            put("data".toByteArray())
            putInt(pcmBytes.toInt())
        }.array()
    }
}
EOF

cat << 'EOF' > app/src/main/java/com/example/myenglishrecorder/VoiceRecorderService.kt
package com.example.myenglishrecorder

import android.annotation.SuppressLint
import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow

class VoiceRecorderService : Service() {

    companion object {
        val liveText = MutableStateFlow("")
        val liveStatus = MutableStateFlow("Ready")
        val isRunning = MutableStateFlow(false)

        @Volatile
        private var engineInstance: VoiceAiEngine? = null

        fun getOrInitEngine(context: Context): VoiceAiEngine {
            return engineInstance ?: synchronized(this) {
                engineInstance ?: VoiceAiEngine(context.applicationContext).also {
                    it.init()
                    engineInstance = it
                }
            }
        }
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var audioRecord: AudioRecord? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private lateinit var storageHelper: AudioStorageHelper

    override fun onCreate() {
        super.onCreate()
        storageHelper = AudioStorageHelper(this)
        getOrInitEngine(applicationContext)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            "STOP" -> {
                stopRecordingAndSave()
                stopSelf()
                return START_NOT_STICKY
            }
            "EXPORT_NOW" -> {
                scope.launch {
                    val res = storageHelper.exportBothToDownloads(clearAfterSave = false)
                    liveStatus.value = formatSavedStatus(res)
                }
                return START_STICKY
            }
        }

        startForegroundWithMicType()
        acquireWakeLock()
        startRecordingLoop()
        return START_STICKY
    }

    private fun startForegroundWithMicType() {
        val channelId = "english_voice_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "English Voice Recorder",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, channelId)
  
