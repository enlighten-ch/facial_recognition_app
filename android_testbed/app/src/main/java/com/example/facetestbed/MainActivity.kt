package com.example.facetestbed

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.ViewFlipper
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    companion object {
        private const val PAGE_MONITOR = 0
        private const val PAGE_REGISTER_CAPTURE = 1
        private const val PAGE_REGISTER_NAME = 2
        private const val PAGE_OUTPUT = 3

        private val REGISTER_DIRECTIONS = listOf("left", "right", "up", "down")
        private val DIRECTION_LABELS = mapOf(
            "left" to "좌",
            "right" to "우",
            "up" to "위",
            "down" to "아래",
        )
        private val DIRECTION_PROMPTS = mapOf(
            "left" to "다음 동작: 얼굴을 화면 기준 왼쪽으로 돌려주세요.",
            "right" to "다음 동작: 얼굴을 화면 기준 오른쪽으로 돌려주세요.",
            "up" to "다음 동작: 턱을 살짝 들고 위를 봐주세요.",
            "down" to "다음 동작: 턱을 살짝 내리고 아래를 봐주세요.",
        )
    }

    private lateinit var headerTitle: TextView
    private lateinit var headerSubtitle: TextView
    private lateinit var backButton: Button
    private lateinit var settingsButton: Button
    private lateinit var quitButton: Button
    private lateinit var previewView: PreviewView
    private lateinit var pageFlipper: ViewFlipper

    private lateinit var monitorStatusText: TextView
    private lateinit var monitorDetailText: TextView
    private lateinit var registerStartButton: Button
    private lateinit var candidateContainer: LinearLayout
    private lateinit var outputNameButton: Button

    private lateinit var registerStatusText: TextView
    private lateinit var registerProgressBar: ProgressBar
    private lateinit var registerCountText: TextView

    private lateinit var registerCompleteText: TextView
    private lateinit var nameInput: EditText
    private lateinit var saveNameButton: Button

    private lateinit var outputLabel: TextView
    private lateinit var outputSubLabel: TextView

    private lateinit var faceDb: FaceDatabase
    private lateinit var faceEngine: FaceRecognitionEngine
    private lateinit var settingsStore: AppSettingsStore
    private var settings = AppSettings()

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val frameInFlight = AtomicBoolean(false)
    private var cameraProvider: ProcessCameraProvider? = null

    private var hasCameraFrame = false
    private var analyzeFrameTick = 0L
    private var registerFrameTick = 0L
    private var lastImageWidth = 0
    private var lastImageHeight = 0

    private var selectedName: String? = null
    private var latestRankings: List<Pair<String, Float>> = emptyList()
    private var lastMonitorStateKey: String? = null

    private var adaptiveLastFrame = -1_000_000_000L
    private var adaptiveUpdateCount = 0

    private val registerDirectionEmbeddings = mutableMapOf(
        "left" to mutableListOf<FloatArray>(),
        "right" to mutableListOf<FloatArray>(),
        "up" to mutableListOf<FloatArray>(),
        "down" to mutableListOf<FloatArray>(),
    )

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else showError("카메라 권한이 필요합니다.")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()

        faceDb = FaceDatabase(File(filesDir, "face_db.json"))
        faceEngine = FaceRecognitionEngine()
        settingsStore = AppSettingsStore(this)
        settings = settingsStore.load()

        registerStartButton.setOnClickListener { startAutoRegistration() }
        outputNameButton.setOnClickListener { outputSelectedName() }
        saveNameButton.setOnClickListener { saveRegistration() }
        backButton.setOnClickListener { goInitialState() }
        settingsButton.setOnClickListener { openSettingsDialog() }
        quitButton.setOnClickListener { finish() }

        goInitialState()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindViews() {
        headerTitle = findViewById(R.id.headerTitle)
        headerSubtitle = findViewById(R.id.headerSubtitle)
        backButton = findViewById(R.id.backButton)
        settingsButton = findViewById(R.id.settingsButton)
        quitButton = findViewById(R.id.quitButton)
        previewView = findViewById(R.id.previewView)
        pageFlipper = findViewById(R.id.pageFlipper)

        monitorStatusText = findViewById(R.id.monitorStatusText)
        monitorDetailText = findViewById(R.id.monitorDetailText)
        registerStartButton = findViewById(R.id.registerStartButton)
        candidateContainer = findViewById(R.id.candidateContainer)
        outputNameButton = findViewById(R.id.outputNameButton)

        registerStatusText = findViewById(R.id.registerStatusText)
        registerProgressBar = findViewById(R.id.registerProgressBar)
        registerCountText = findViewById(R.id.registerCountText)

        registerCompleteText = findViewById(R.id.registerCompleteText)
        nameInput = findViewById(R.id.nameInput)
        saveNameButton = findViewById(R.id.saveNameButton)

        outputLabel = findViewById(R.id.outputLabel)
        outputSubLabel = findViewById(R.id.outputSubLabel)
    }

    private fun currentPage(): Int = pageFlipper.displayedChild

    private fun goTo(page: Int) {
        pageFlipper.displayedChild = page
        when (page) {
            PAGE_MONITOR -> {
                headerTitle.text = "얼굴 인식"
                headerSubtitle.text = "카메라를 보고 있으면 자동으로 등록/인식이 진행됩니다."
                backButton.visibility = Button.GONE
            }
            PAGE_REGISTER_CAPTURE -> {
                headerTitle.text = "얼굴 등록"
                headerSubtitle.text = "랜드마크 기반으로 좌/우/위/아래 샘플을 자동 수집합니다."
                backButton.visibility = Button.VISIBLE
            }
            PAGE_REGISTER_NAME -> {
                headerTitle.text = "이름 저장"
                headerSubtitle.text = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
                backButton.visibility = Button.VISIBLE
            }
            PAGE_OUTPUT -> {
                headerTitle.text = "이름 출력"
                headerSubtitle.text = "선택된 이름을 출력합니다."
                backButton.visibility = Button.VISIBLE
            }
        }
    }

    private fun registrationTotalTarget(): Int {
        return settings.registerSamplesPerDirection.coerceAtLeast(1) * REGISTER_DIRECTIONS.size
    }

    private fun resetRegistrationState() {
        REGISTER_DIRECTIONS.forEach { registerDirectionEmbeddings[it]?.clear() }
        registerFrameTick = 0
        setRegisterCount()
    }

    private fun totalRegisteredSamples(): Int {
        return REGISTER_DIRECTIONS.sumOf { registerDirectionEmbeddings[it]?.size ?: 0 }
    }

    private fun setRegisterCount() {
        val total = registrationTotalTarget()
        val current = totalRegisteredSamples()
        registerProgressBar.max = total
        registerProgressBar.progress = current.coerceAtMost(total)
        registerCountText.text = "$current / $total"
    }

    private fun nextTargetDirection(): String? {
        val need = settings.registerSamplesPerDirection.coerceAtLeast(1)
        for (dir in REGISTER_DIRECTIONS) {
            if ((registerDirectionEmbeddings[dir]?.size ?: 0) < need) return dir
        }
        return null
    }

    private fun summarizeDirectionProgress(): String {
        val need = settings.registerSamplesPerDirection.coerceAtLeast(1)
        return REGISTER_DIRECTIONS.joinToString(" | ") { dir ->
            val c = registerDirectionEmbeddings[dir]?.size ?: 0
            val label = DIRECTION_LABELS[dir] ?: dir
            "$label:$c/$need"
        }
    }

    private fun allRegisteredEmbeddings(): List<FloatArray> {
        val out = mutableListOf<FloatArray>()
        REGISTER_DIRECTIONS.forEach { out.addAll(registerDirectionEmbeddings[it] ?: emptyList()) }
        return out
    }

    private fun goInitialState() {
        resetRegistrationState()
        selectedName = null
        latestRankings = emptyList()
        lastMonitorStateKey = null
        adaptiveLastFrame = -1_000_000_000L
        adaptiveUpdateCount = 0
        nameInput.setText("")
        applyMonitorIdleState()
        goTo(PAGE_MONITOR)
    }

    private fun clearCandidates() {
        candidateContainer.removeAllViews()
    }

    private fun applyMonitorIdleState() {
        monitorStatusText.text = "환영해요 :) 얼굴 등록 및 인식을 위해 앞을 똑바로 바라봐주세요"
        monitorDetailText.text = ""
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        outputNameButton.isEnabled = false
        clearCandidates()
    }

    private fun applyUnregisteredState() {
        monitorStatusText.text = "등록이 되어있지 않아요! 등록할까요?"
        monitorDetailText.text = "등록시작 버튼을 누르면 좌/우/위/아래 방향별로 ${settings.registerSamplesPerDirection}장씩 수집합니다."
        registerStartButton.visibility = Button.VISIBLE
        registerStartButton.isEnabled = true
        outputNameButton.isEnabled = false
        clearCandidates()
    }

    private fun applyRegisteredClearState(topName: String, topScore: Float, margin: Float) {
        monitorStatusText.text = "등록된 사용자입니다: $topName"
        monitorDetailText.text = String.format(Locale.US, "1위 점수 %.3f, 2위와 차이 %.3f", topScore, margin)
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        clearCandidates()

        val btn = Button(this).apply {
            text = "1위 $topName"
            isEnabled = false
        }
        candidateContainer.addView(btn)

        selectedName = topName
        outputNameButton.isEnabled = true
    }

    private fun applyRegisteredAmbiguousState(rankings: List<Pair<String, Float>>) {
        monitorStatusText.text = "유사한 얼굴이 있습니다. 이름을 선택해 주세요."
        monitorDetailText.text = "1~3위 후보 중 이름 선택 후 이름 출력하기 버튼이 활성화됩니다."
        registerStartButton.visibility = Button.GONE
        registerStartButton.isEnabled = false
        clearCandidates()
        selectedName = null
        outputNameButton.isEnabled = false

        rankings.forEachIndexed { idx, (name, score) ->
            val btn = Button(this).apply {
                text = String.format(Locale.US, "%d위 선택: %s (점수 %.3f)", idx + 1, name, score)
                setOnClickListener { chooseCandidateName(name) }
            }
            candidateContainer.addView(btn)
        }
    }

    private fun chooseCandidateName(name: String) {
        selectedName = name
        monitorDetailText.text = "선택됨: $name / 이름 출력하기 버튼을 눌러주세요."
        outputNameButton.isEnabled = true
    }

    private fun outputSelectedName() {
        val name = selectedName
        if (name.isNullOrBlank()) {
            showError("먼저 이름을 선택해 주세요.")
            return
        }
        outputLabel.text = name
        outputSubLabel.text = if (settings.printEnabled) {
            "프린터로 출력 전송됨: $name"
        } else {
            "이름이 출력되었습니다."
        }
        goTo(PAGE_OUTPUT)
    }

    private fun startAutoRegistration() {
        if (!hasCameraFrame) {
            showError("카메라 프레임이 아직 없습니다.")
            return
        }
        resetRegistrationState()
        val firstDir = nextTargetDirection()
        val firstPrompt = if (firstDir != null) DIRECTION_PROMPTS[firstDir] ?: "" else ""
        registerStatusText.text = "$firstPrompt\n진행: ${summarizeDirectionProgress()}"
        goTo(PAGE_REGISTER_CAPTURE)
    }

    private fun evaluateSimilarityBand(embedding: FloatArray): Pair<Boolean, String> {
        val existing = allRegisteredEmbeddings()
        if (existing.isEmpty()) return true to ""

        val maxSim = existing.maxOf { dot(it, embedding) }
        if (maxSim > settings.registerSimilarityHigherBoundary) {
            return false to String.format(Locale.US, "중복 샘플입니다. 조금 더 다른 각도로 움직여주세요. (max=%.3f)", maxSim)
        }
        if (maxSim < settings.registerSimilarityLowerBoundary) {
            return false to String.format(Locale.US, "유사도가 낮습니다. 같은 사람이 카메라 앞에 있어야 합니다. (max=%.3f)", maxSim)
        }
        return true to String.format(Locale.US, "유사도 적합 구간 (max=%.3f)", maxSim)
    }

    private fun handleRegisterCapture(face: Face, embedding: FloatArray) {
        registerFrameTick += 1
        if (registerFrameTick % settings.registerCaptureEveryNFrames.coerceAtLeast(1) != 0L) return

        val targetDirection = nextTargetDirection()
        if (targetDirection == null) {
            registerStatusText.text = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
            goTo(PAGE_REGISTER_NAME)
            return
        }

        val center = face.boundingBox.centerX() to face.boundingBox.centerY()
        val ellipse = getRegisterEllipse(lastImageWidth, lastImageHeight)
        if (!isInsideEllipse(center.first, center.second, ellipse.first.first, ellipse.first.second, ellipse.second.first, ellipse.second.second)) {
            registerStatusText.text = "얼굴을 타원 가이드 안으로 맞춰주세요.\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}"
            return
        }

        val detectedDirection = faceEngine.classifyDirection(face)
        if (detectedDirection != targetDirection) {
            val detectedLabel = DIRECTION_LABELS[detectedDirection] ?: "정면"
            registerStatusText.text =
                "현재 감지 방향: $detectedLabel\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}"
            return
        }

        val (accepted, msg) = evaluateSimilarityBand(embedding)
        if (!accepted) {
            registerStatusText.text = "$msg\n${DIRECTION_PROMPTS[targetDirection]}\n진행: ${summarizeDirectionProgress()}"
            return
        }

        registerDirectionEmbeddings[targetDirection]?.add(embedding)
        setRegisterCount()

        val nextDirection = nextTargetDirection()
        if (nextDirection == null) {
            registerStatusText.text = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
            goTo(PAGE_REGISTER_NAME)
            return
        }

        registerStatusText.text = "샘플 저장 완료. $msg\n${DIRECTION_PROMPTS[nextDirection]}\n진행: ${summarizeDirectionProgress()}"
    }

    private fun saveRegistration() {
        val name = nameInput.text?.toString()?.trim().orEmpty()
        if (name.isBlank()) {
            showError("이름을 입력해 주세요.")
            return
        }

        val embeddings = allRegisteredEmbeddings()
        if (embeddings.size < registrationTotalTarget()) {
            showError("얼굴 샘플이 아직 충분하지 않습니다.")
            return
        }

        faceDb.upsertPerson(name, embeddings, settings.maxEmbeddingsPerPerson)
        goInitialState()
    }

    private fun tryAdaptiveUpdate(personName: String, topScore: Float, embedding: FloatArray) {
        if (!settings.adaptiveUpdateEnabled) return
        if (topScore < settings.authSimThreshold || topScore > settings.adaptiveUpdateMaxSim) return
        if (adaptiveUpdateCount >= settings.adaptiveUpdateMaxSamplesPerSession) return
        if ((analyzeFrameTick - adaptiveLastFrame) < settings.adaptiveUpdateMinIntervalFrames) return

        if (faceDb.appendSampleToPerson(personName, embedding, settings.maxEmbeddingsPerPerson)) {
            adaptiveLastFrame = analyzeFrameTick
            adaptiveUpdateCount += 1
            monitorDetailText.text = "동일 인물의 얼굴 변화가 감지되었습니다. 데이터를 업데이트 합니다."
        }
    }

    private fun applyMonitorStateFromFace(face: Face, embedding: FloatArray) {
        if (settings.recognitionUseEllipseMask) {
            val center = face.boundingBox.centerX() to face.boundingBox.centerY()
            val ellipse = getRegisterEllipse(lastImageWidth, lastImageHeight)
            if (!isInsideEllipse(center.first, center.second, ellipse.first.first, ellipse.first.second, ellipse.second.first, ellipse.second.second)) {
                val stateKey = "idle:outside"
                if (stateKey != lastMonitorStateKey) {
                    applyMonitorIdleState()
                    lastMonitorStateKey = stateKey
                }
                return
            }
        }

        val rankings = faceDb.rank(embedding, settings.topK.coerceAtLeast(1))
        latestRankings = rankings

        if (rankings.isEmpty()) {
            val stateKey = "unknown:none"
            if (stateKey != lastMonitorStateKey) {
                applyUnregisteredState()
                lastMonitorStateKey = stateKey
            }
            return
        }

        val topName = rankings[0].first
        val topScore = rankings[0].second
        if (topScore < settings.authSimThreshold) {
            val stateKey = "unknown:score:${String.format(Locale.US, "%.3f", topScore)}"
            if (stateKey != lastMonitorStateKey) {
                applyUnregisteredState()
                lastMonitorStateKey = stateKey
            }
            return
        }

        val secondScore = rankings.getOrNull(1)?.second ?: -1f
        val margin = if (secondScore >= 0f) topScore - secondScore else topScore

        if (margin >= settings.authMarginThreshold) {
            tryAdaptiveUpdate(topName, topScore, embedding)
            val stateKey = "known:clear:$topName:${String.format(Locale.US, "%.3f", topScore)}:${String.format(Locale.US, "%.3f", margin)}"
            if (stateKey != lastMonitorStateKey) {
                applyRegisteredClearState(topName, topScore, margin)
                lastMonitorStateKey = stateKey
            }
            return
        }

        val top3 = rankings.take(3)
        val stateKey = "known:ambiguous:" + top3.joinToString("|") { "${it.first}:${String.format(Locale.US, "%.3f", it.second)}" }
        if (stateKey != lastMonitorStateKey) {
            applyRegisteredAmbiguousState(top3)
            lastMonitorStateKey = stateKey
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = try {
                cameraProviderFuture.get()
            } catch (t: Throwable) {
                showError("카메라 준비 실패: ${t.javaClass.simpleName}")
                return@addListener
            }

            cameraProvider = provider
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { image ->
                if (frameInFlight.getAndSet(true)) {
                    image.close()
                    return@setAnalyzer
                }
                analyzeFrame(image)
            }

            val selector = pickCameraSelector(provider)
            if (selector == null) {
                frameInFlight.set(false)
                showError("카메라를 찾을 수 없습니다. 설정에서 CAMERA_INDEX를 바꿔보세요.")
                return@addListener
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (t: Throwable) {
                showError("카메라 바인딩 실패: ${t.javaClass.simpleName}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun restartCamera() {
        cameraProvider?.unbindAll()
        frameInFlight.set(false)
        startCamera()
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        hasCameraFrame = true
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            frameInFlight.set(false)
            imageProxy.close()
            return
        }

        val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        lastImageWidth = inputImage.width
        lastImageHeight = inputImage.height

        faceEngine.detectLargestFace(inputImage)
            .addOnSuccessListener { faces ->
                val face = pickLargestFace(faces)
                val page = currentPage()
                if (face == null) {
                    if (page == PAGE_MONITOR) {
                        val stateKey = "idle"
                        if (stateKey != lastMonitorStateKey) {
                            runOnUiThread { applyMonitorIdleState() }
                            lastMonitorStateKey = stateKey
                        }
                    } else if (page == PAGE_REGISTER_CAPTURE) {
                        runOnUiThread {
                            val t = nextTargetDirection()
                            if (t != null) {
                                registerStatusText.text = "얼굴을 찾지 못했습니다. 화면 중앙으로 와주세요.\n${DIRECTION_PROMPTS[t]}\n진행: ${summarizeDirectionProgress()}"
                            }
                        }
                    }
                    return@addOnSuccessListener
                }

                val embedding = faceEngine.makeEmbedding(face, inputImage.width, inputImage.height)
                when (page) {
                    PAGE_MONITOR -> {
                        analyzeFrameTick += 1
                        if (analyzeFrameTick % settings.analyzeEveryNFrames.coerceAtLeast(1) == 0L) {
                            runOnUiThread { applyMonitorStateFromFace(face, embedding) }
                        }
                    }
                    PAGE_REGISTER_CAPTURE -> {
                        runOnUiThread { handleRegisterCapture(face, embedding) }
                    }
                }
            }
            .addOnFailureListener { err ->
                runOnUiThread { showError("얼굴 분석 실패: ${err.javaClass.simpleName}") }
            }
            .addOnCompleteListener {
                frameInFlight.set(false)
                imageProxy.close()
            }
    }

    private fun pickLargestFace(faces: List<Face>): Face? {
        if (faces.isEmpty()) return null
        return faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
    }

    private fun pickCameraSelector(provider: ProcessCameraProvider): CameraSelector? {
        val preferred = if (settings.cameraIndex == 1) {
            listOf(CameraSelector.DEFAULT_FRONT_CAMERA, CameraSelector.DEFAULT_BACK_CAMERA)
        } else {
            listOf(CameraSelector.DEFAULT_BACK_CAMERA, CameraSelector.DEFAULT_FRONT_CAMERA)
        }
        for (selector in preferred) {
            try {
                if (provider.hasCamera(selector)) return selector
            } catch (_: Throwable) {
            }
        }
        return null
    }

    private fun getRegisterEllipse(width: Int, height: Int): Pair<Pair<Int, Int>, Pair<Int, Int>> {
        val w = width.coerceAtLeast(1)
        val h = height.coerceAtLeast(1)
        val cx = w / 2
        val cy = (h * (0.5 + settings.registerEllipseCenterYOffsetRatio)).toInt()
        val ax = (w * settings.registerEllipseAxisXRatio).toInt().coerceAtLeast(40)
        val ay = (h * settings.registerEllipseAxisYRatio).toInt().coerceAtLeast(60)
        return (cx to cy) to (ax to ay)
    }

    private fun isInsideEllipse(x: Int, y: Int, cx: Int, cy: Int, ax: Int, ay: Int): Boolean {
        if (ax <= 0 || ay <= 0) return false
        val dx = (x - cx) / ax.toDouble()
        val dy = (y - cy) / ay.toDouble()
        return dx * dx + dy * dy <= 1.0
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        val n = minOf(a.size, b.size)
        var sum = 0f
        for (i in 0 until n) sum += a[i] * b[i]
        return sum
    }

    private fun openSettingsDialog() {
        SettingsDialog.show(this, settings) { updated ->
            settings = updated
            settingsStore.save(updated)
            setRegisterCount()
            resetRegistrationState()
            goInitialState()
            restartCamera()
        }
    }

    private fun showError(message: String) {
        AlertDialog.Builder(this)
            .setTitle("오류")
            .setMessage(message)
            .setPositiveButton("확인", null)
            .show()
    }

    override fun onDestroy() {
        cameraProvider?.unbindAll()
        faceEngine.close()
        cameraExecutor.shutdown()
        super.onDestroy()
    }
}
