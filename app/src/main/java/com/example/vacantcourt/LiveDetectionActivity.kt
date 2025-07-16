package com.example.vacantcourt

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.updateLayoutParams
import com.example.vacantcourt.data.PointData
import com.example.vacantcourt.data.TennisComplexData
import com.example.vacantcourt.databinding.ActivityMainBinding
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class DetectionResult(
    val boundingBox: RectF,
    val className: String,
    val confidence: Float
)

class LiveDetectionActivity : ComponentActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var tfliteInterpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var preview: Preview? = null

    private var complexId: String? = null
    private var normalizedCourtPolygons: List<Pair<String, List<PointData>>> = emptyList()
    private val db = Firebase.firestore
    private val TENNIS_COMPLEXES_COLLECTION_LIVE = "Courts"

    private val TAG = "LiveDetectionActivity"
    private val CAMERA_PERMISSION_REQUEST_CODE = 100

    private val MODEL_NAME = "yolov5.tflite"
    private val LABELS_NAME = "coco_labels.txt"
    private val CONFIDENCE_THRESHOLD = 0.4f
    private val IOU_THRESHOLD = 0.5f
    private val MAX_DETECTED_OBJECTS = 10

    private var modelInputWidth = 0
    private var modelInputHeight = 0
    private var modelOutputClasses = 0

    private val interpreterLock = Any()
    @Volatile
    private var processingEnabled = true

    private lateinit var textViewComplexNameTitleLive: TextView

    private val courtStatusMap: MutableMap<String, String> = mutableMapOf()
    private val courtPendingFirebaseUpdate: MutableSet<String> = mutableSetOf()

    private var lastInferenceTimestamp: Long = 0L
    private val INFERENCE_INTERVAL_MS = 5000L

    private var lastHeartbeatTimestamp: Long = 0L
    private val HEARTBEAT_INTERVAL_MS = 15000L // 15 seconds


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        processingEnabled = true
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        textViewComplexNameTitleLive = binding.textViewComplexNameTitleLive

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            WindowInsetsControllerCompat(window, binding.root).let { controller ->
                controller.hide(WindowInsetsCompat.Type.systemBars())
                controller.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                    View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                            or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            or View.SYSTEM_UI_FLAG_FULLSCREEN)
        }
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

        applyWindowInsets()

        complexId = intent.getStringExtra("COMPLEX_ID")
        val complexName = intent.getStringExtra("COMPLEX_NAME")

        textViewComplexNameTitleLive.text = complexName ?: getString(R.string.configure_courts_activity_title)


        if (complexId == null) {
            Toast.makeText(this, "Error: Complex ID missing.", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            loadCourtConfigurationAndStart()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun applyWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { _, windowInsets ->
            textViewComplexNameTitleLive.updateLayoutParams<ViewGroup.MarginLayoutParams> {}
            WindowInsetsCompat.CONSUMED
        }
    }

    private fun loadCourtConfigurationAndStart() {
        complexId?.let { id ->
            db.collection(TENNIS_COMPLEXES_COLLECTION_LIVE).document(id)
                .get()
                .addOnSuccessListener { document ->
                    if (!processingEnabled || isDestroyed || isFinishing) return@addOnSuccessListener
                    if (document.exists()) {
                        val complex = document.toObject(TennisComplexData::class.java)
                        complex?.let { loadedComplex ->
                            if (textViewComplexNameTitleLive.text == getString(R.string.configure_courts_activity_title) && loadedComplex.name.isNotEmpty()) {
                                textViewComplexNameTitleLive.text = loadedComplex.name
                            }
                            courtStatusMap.clear()
                            courtPendingFirebaseUpdate.clear()
                            loadedComplex.courts.forEach { court ->
                                courtStatusMap[court.name] = court.status
                            }

                            normalizedCourtPolygons = loadedComplex.courts.filter { court ->
                                court.isConfigured && court.regionPoints != null && court.regionPoints!!.isNotEmpty()
                            }.map { configuredCourt ->
                                Pair(configuredCourt.name, configuredCourt.regionPoints!!)
                            }
                            if (normalizedCourtPolygons.isEmpty()) {
                                Toast.makeText(this, getString(R.string.no_courts_configured_for_viewing_live), Toast.LENGTH_LONG).show()
                                finish()
                                return@addOnSuccessListener
                            }
                            Log.d(TAG, "Loaded ${normalizedCourtPolygons.size} configured regions. Initial statuses: $courtStatusMap")
                            setupYoloInterpreter()
                            startCameraWhenReady()
                            lastInferenceTimestamp = 0L
                            lastHeartbeatTimestamp = 0L

                        } ?: run {
                            showErrorAndFinish(getString(R.string.error_parsing_complex_data_live))
                        }
                    } else {
                        showErrorAndFinish(getString(R.string.complex_not_found_live))
                    }
                }
                .addOnFailureListener { e ->
                    if (!processingEnabled || isDestroyed || isFinishing) return@addOnFailureListener
                    showErrorAndFinish(getString(R.string.error_fetching_config_live, e.localizedMessage ?: "Unknown error"))
                }
        }
    }
    private fun showErrorAndFinish(message: String) {
        if (isFinishing || isDestroyed) return
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        Log.e(TAG, message)
        finish()
    }

    private fun setupYoloInterpreter() {
        synchronized(interpreterLock) {
            if (tfliteInterpreter != null) {
                Log.d(TAG, "Interpreter already initialized.")
                return
            }
            try {
                val model = FileUtil.loadMappedFile(this, MODEL_NAME)
                labels = FileUtil.loadLabels(this, LABELS_NAME)
                val interpreterOptions = Interpreter.Options()
                tfliteInterpreter = Interpreter(model, interpreterOptions)

                val inputTensor = tfliteInterpreter!!.getInputTensor(0)
                val inputShape = inputTensor.shape()
                modelInputHeight = inputShape[1]
                modelInputWidth = inputShape[2]

                val outputTensor = tfliteInterpreter!!.getOutputTensor(0)
                val outputShape = outputTensor.shape()
                modelOutputClasses = outputShape[outputShape.size - 1] - 5

                Log.d(TAG, "YOLOv5 Interpreter Initialized. Input: $modelInputWidth x $modelInputHeight, Output Classes: $modelOutputClasses, Output Shape: ${outputShape.joinToString()}")

            } catch (e: IOException) {
                Log.e(TAG, "Error initializing TFLite interpreter: ${e.message}", e)
                if (processingEnabled && !isFinishing && !isDestroyed) {
                    showErrorAndFinish(getString(R.string.error_loading_model_live, e.localizedMessage ?: "IOException"))
                }
            }
        }
    }

    private fun startCameraWhenReady() {
        binding.previewView.post {
            if (isDestroyed || isFinishing || !processingEnabled) {
                Log.w(TAG, "Activity is Destroyed/Finishing or processing disabled before camera could start.")
                return@post
            }
            Log.d(TAG, "PreviewView is ready, proceeding to startCamera.")
            startCamera()
        }
    }

    @SuppressLint("SetTextI18n")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            if (!processingEnabled || isDestroyed || isFinishing) return@addListener
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get camera provider or bind use cases", e)
                if (processingEnabled && !isFinishing && !isDestroyed) {
                    showErrorAndFinish("Camera setup failed: ${e.localizedMessage}")
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val currentCameraProvider = cameraProvider ?: run {
            Log.e(TAG, "Camera provider not available.")
            if (processingEnabled && !isFinishing && !isDestroyed) {
                showErrorAndFinish("Camera provider initialization failed.")
            }
            return
        }

        if (binding.previewView.display == null) {
            Log.e(TAG, "PreviewView display is null, cannot get rotation. Retrying soon.")
            if (processingEnabled && !isDestroyed && !isFinishing) {
                binding.previewView.postDelayed({
                    if (processingEnabled && !isDestroyed && !isFinishing) bindCameraUseCases()
                },100)
            }
            return
        }

        val previewViewWidth = binding.previewView.width
        val previewViewHeight = binding.previewView.height

        if (previewViewWidth == 0 || previewViewHeight == 0) {
            Log.w(TAG, "PreviewView dimensions are zero ($previewViewWidth x $previewViewHeight). Postponing camera binding.")
            if (processingEnabled && !isDestroyed && !isFinishing) {
                binding.previewView.post {
                    if (processingEnabled && !isDestroyed && !isFinishing) bindCameraUseCases()
                }
            }
            return
        }

        val screenAspectRatio = aspectRatio(previewViewWidth, previewViewHeight)
        val rotation = binding.previewView.display.rotation

        preview = Preview.Builder()
            .setTargetAspectRatio(screenAspectRatio)
            .setTargetRotation(rotation)
            .build()
            .also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(screenAspectRatio)
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, YoloImageAnalyzer())
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            currentCameraProvider.unbindAll()
            currentCameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )
            Log.d(TAG, "Camera use cases bound.")
            if (processingEnabled && !isDestroyed && !isFinishing) {
                binding.overlayView.setConfiguredRegions(normalizedCourtPolygons, previewViewWidth, previewViewHeight)
            }
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
            if (processingEnabled && !isFinishing && !isDestroyed) {
                Toast.makeText(this, "Use case binding failed: ${exc.localizedMessage}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun aspectRatio(width: Int, height: Int): Int {
        val previewRatio = max(width, height).toDouble() / min(width, height)
        return if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
            AspectRatio.RATIO_4_3
        } else {
            AspectRatio.RATIO_16_9
        }
    }

    private inner class YoloImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            val currentTime = System.currentTimeMillis()

            if(currentTime - lastHeartbeatTimestamp > HEARTBEAT_INTERVAL_MS) {
                updateComplexHeartbeat()
                lastHeartbeatTimestamp = currentTime
            }

            if (currentTime - lastInferenceTimestamp < INFERENCE_INTERVAL_MS) {
                imageProxy.close()
                return
            }
            lastInferenceTimestamp = currentTime
            Log.d(TAG, "Performing inference. Time since last: ${currentTime - (lastInferenceTimestamp - INFERENCE_INTERVAL_MS)} ms")


            if (!processingEnabled || tfliteInterpreter == null || labels.isEmpty() || modelInputWidth == 0 || normalizedCourtPolygons.isEmpty()) {
                imageProxy.close()
                return
            }

            val uprightBitmapForOverlay: Bitmap
            val inputBuffer: ByteBuffer
            val outputBuffer: TensorBuffer

            try {
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                val sourceBitmap = imageProxy.toBitmap() ?: run {
                    Log.e(TAG, "Failed to convert ImageProxy to Bitmap.")
                    imageProxy.close()
                    return
                }

                uprightBitmapForOverlay = sourceBitmap.rotate(rotationDegrees.toFloat()) ?: run {
                    Log.e(TAG, "Failed to create upright bitmap for overlay.")
                    imageProxy.close()
                    return
                }

                val currentInterpreterInstance: Interpreter?
                synchronized(interpreterLock) {
                    if (!processingEnabled || tfliteInterpreter == null) {
                        imageProxy.close()
                        return
                    }
                    currentInterpreterInstance = tfliteInterpreter

                    val inputTensorModelSpec = currentInterpreterInstance!!.getInputTensor(0)
                    val imageProcessorForModel = ImageProcessor.Builder()
                        .add(Rot90Op(-rotationDegrees / 90))
                        .add(ResizeOp(modelInputHeight, modelInputWidth, ResizeOp.ResizeMethod.BILINEAR))
                        .add(org.tensorflow.lite.support.common.ops.NormalizeOp(0f, 255f))
                        .build()

                    var tensorImageForModel = TensorImage(inputTensorModelSpec.dataType())
                    tensorImageForModel.load(sourceBitmap)
                    tensorImageForModel = imageProcessorForModel.process(tensorImageForModel)

                    inputBuffer = tensorImageForModel.buffer
                    inputBuffer.rewind()

                    val outputTensor = currentInterpreterInstance!!.getOutputTensor(0)
                    outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())
                    outputBuffer.buffer.rewind()

                    currentInterpreterInstance!!.run(inputBuffer, outputBuffer.buffer.rewind())
                }

                val detectionResults = processYoloOutput(outputBuffer, uprightBitmapForOverlay.width, uprightBitmapForOverlay.height)
                updateCourtStatusesBasedOnLastInference(detectionResults, uprightBitmapForOverlay.width, uprightBitmapForOverlay.height)


                runOnUiThread {
                    if (!processingEnabled || isDestroyed || isFinishing) return@runOnUiThread
                    binding.overlayView.setResults(
                        detectionResults,
                        uprightBitmapForOverlay.height,
                        uprightBitmapForOverlay.width
                    )
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error during YOLO inference or post-processing: ${e.message}", e)
            } finally {
                imageProxy.close()
            }
        }
    }

    private fun updateComplexHeartbeat() {
        val complexDocRef = complexId?.let { db.collection(TENNIS_COMPLEXES_COLLECTION_LIVE).document(it) }
            ?: return

        val updateData = mapOf("lastUpdatedStatus" to System.currentTimeMillis())
        complexDocRef.update(updateData)
            .addOnSuccessListener { Log.d(TAG, "Complex heartbeat updated successfully.") }
            .addOnFailureListener { e -> Log.w(TAG, "Error updating complex heartbeat", e) }
    }


    private fun isPointInPolygon(point: PointF, polygon: List<PointF>): Boolean {
        if (polygon.size < 3) return false
        var crossings = 0
        for (i in polygon.indices) {
            val p1 = polygon[i]
            val p2 = polygon[(i + 1) % polygon.size]

            if (((p1.y <= point.y && point.y < p2.y) || (p2.y <= point.y && point.y < p1.y)) &&
                (point.x < (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x)
            ) {
                crossings++
            }
        }
        return crossings % 2 == 1
    }

    private fun checkOverlap(personRect: RectF, courtPolygonAbsolute: List<PointF>): Boolean {
        if (courtPolygonAbsolute.isEmpty()) return false

        val personPoints = listOf(
            PointF(personRect.left, personRect.top),
            PointF(personRect.right, personRect.top),
            PointF(personRect.right, personRect.bottom),
            PointF(personRect.left, personRect.bottom)
        )

        for (personPoint in personPoints) {
            if (isPointInPolygon(personPoint, courtPolygonAbsolute)) return true
        }

        for (courtPoint in courtPolygonAbsolute) {
            if (personRect.contains(courtPoint.x, courtPoint.y)) return true
        }

        val personPath = Path().apply {
            moveTo(personRect.left, personRect.top)
            lineTo(personRect.right, personRect.top)
            lineTo(personRect.right, personRect.bottom)
            lineTo(personRect.left, personRect.bottom)
            close()
        }
        val courtPath = Path().apply {
            if (courtPolygonAbsolute.isNotEmpty()) {
                moveTo(courtPolygonAbsolute[0].x, courtPolygonAbsolute[0].y)
                for (i in 1 until courtPolygonAbsolute.size) {
                    lineTo(courtPolygonAbsolute[i].x, courtPolygonAbsolute[i].y)
                }
                close()
            }
        }

        val regionPerson = Region()
        regionPerson.setPath(personPath, Region(0,0, Int.MAX_VALUE, Int.MAX_VALUE))
        val regionCourt = Region()
        regionCourt.setPath(courtPath, Region(0,0, Int.MAX_VALUE, Int.MAX_VALUE))

        return regionPerson.op(regionCourt, Region.Op.INTERSECT)
    }


    private fun updateCourtStatusesBasedOnLastInference(
        detectionResults: List<DetectionResult>,
        imageWidth: Int,
        imageHeight: Int
    ) {
        if (imageWidth == 0 || imageHeight == 0 || complexId == null) return

        val personDetections = detectionResults.filter { it.className.equals("person", ignoreCase = true) }

        for ((courtName, normalizedPolygonPoints) in normalizedCourtPolygons) {
            if (courtPendingFirebaseUpdate.contains(courtName)) {
                Log.d(TAG, "Skipping status update for '$courtName', Firebase update pending.")
                continue
            }

            val absolutePolygonPoints = normalizedPolygonPoints.map {
                PointF(it.x * imageWidth, it.y * imageHeight)
            }

            var isCourtOccupiedThisInference = false
            if (personDetections.isNotEmpty()) {
                for (personDetection in personDetections) {
                    if (checkOverlap(personDetection.boundingBox, absolutePolygonPoints)) {
                        isCourtOccupiedThisInference = true
                        break
                    }
                }
            }

            val currentLocalStatus = courtStatusMap[courtName] ?: "available"
            val newStatus = if (isCourtOccupiedThisInference) "in-use" else "available"

            if (currentLocalStatus != newStatus) {
                Log.d(TAG, "Court '$courtName' status changed. Current local: '$currentLocalStatus', New based on inference: '$newStatus'. Attempting update.")
                attemptFirebaseUpdate(courtName, newStatus)
            } else {
                Log.v(TAG, "Court '$courtName' status ('$currentLocalStatus') consistent with inference. No change needed.")
            }
        }
    }

    private fun attemptFirebaseUpdate(courtName: String, newStatus: String) {
        if (courtPendingFirebaseUpdate.contains(courtName)) {
            Log.d(TAG, "Update for '$courtName' already pending. Skipping.")
            return
        }
        val currentLocalStatus = courtStatusMap[courtName]
        if (currentLocalStatus == newStatus) {
            Log.d(TAG, "Court '$courtName' local status already '$newStatus'. No Firebase update needed.")
            return
        }

        courtPendingFirebaseUpdate.add(courtName)
        Log.d(TAG, "Attempting Firebase update for '$courtName' to '$newStatus'. currentLocalStatus: $currentLocalStatus")

        val complexDocRef = complexId?.let { db.collection(TENNIS_COMPLEXES_COLLECTION_LIVE).document(it) }
            ?: run {
                Log.e(TAG, "Complex ID is null, cannot update status for $courtName")
                courtPendingFirebaseUpdate.remove(courtName)
                return
            }

        db.runTransaction { transaction ->
            val snapshot = transaction.get(complexDocRef)
            val currentComplexData = snapshot.toObject(TennisComplexData::class.java)
            var changedInTransaction = false

            currentComplexData?.let { complexData ->
                val courtIndex = complexData.courts.indexOfFirst { it.name == courtName }
                if (courtIndex != -1) {
                    val courtToUpdate = complexData.courts[courtIndex]
                    if (courtToUpdate.status != newStatus) {
                        val updatedCourt = courtToUpdate.copy(status = newStatus, lastUpdatedStatus = System.currentTimeMillis())
                        val mutableCourts = complexData.courts.toMutableList()
                        mutableCourts[courtIndex] = updatedCourt
                        transaction.update(complexDocRef, "courts", mutableCourts.toList())
                        changedInTransaction = true
                        Log.i(TAG, "Firebase: Transaction will update court '$courtName' from '${courtToUpdate.status}' to '$newStatus'.")
                    } else {
                        Log.d(TAG, "Firebase: Court '$courtName' status in DB already '$newStatus' during transaction read. No DB write needed.")
                    }
                } else {
                    Log.e(TAG, "Court '$courtName' not found in complex data during transaction.")
                }
            } ?: Log.e(TAG, "Complex data not found in transaction for $complexId")
            changedInTransaction
        }.addOnSuccessListener { changedInDb ->
            Log.d(TAG, "Transaction for '$courtName' to '$newStatus' success. Changed in DB: $changedInDb")
            runOnUiThread {
                courtStatusMap[courtName] = newStatus
                courtPendingFirebaseUpdate.remove(courtName)
            }
        }.addOnFailureListener { e ->
            Log.e(TAG, "Transaction for '$courtName' to '$newStatus' FAILED: $e")
            runOnUiThread {
                courtPendingFirebaseUpdate.remove(courtName)
            }
        }
    }


    private fun processYoloOutput(outputBuffer: TensorBuffer, imageToScaleCoordsToWidth: Int, imageToScaleCoordsToHeight: Int): List<DetectionResult> {
        val outputArray = outputBuffer.floatArray
        val numPredictions = tfliteInterpreter!!.getOutputTensor(0).shape()[1]
        val numElementsPerPrediction = modelOutputClasses + 5
        val candidates = mutableListOf<DetectionResult>()

        for (i in 0 until numPredictions) {
            val offset = i * numElementsPerPrediction
            val confidence = outputArray[offset + 4]

            if (confidence >= CONFIDENCE_THRESHOLD) {
                var maxClassScore = 0f
                var classId = -1
                for (c in 0 until modelOutputClasses) {
                    val classScore = outputArray[offset + 5 + c]
                    if (classScore > maxClassScore) {
                        maxClassScore = classScore
                        classId = c
                    }
                }

                if (classId != -1 && maxClassScore > CONFIDENCE_THRESHOLD) {
                    val xCenter = outputArray[offset + 0] * imageToScaleCoordsToWidth
                    val yCenter = outputArray[offset + 1] * imageToScaleCoordsToHeight
                    val width = outputArray[offset + 2] * imageToScaleCoordsToWidth
                    val height = outputArray[offset + 3] * imageToScaleCoordsToHeight

                    val left = xCenter - width / 2
                    val top = yCenter - height / 2
                    val right = xCenter + width / 2
                    val bottom = yCenter + height / 2

                    val clampedLeft = max(0f, left)
                    val clampedTop = max(0f, top)
                    val clampedRight = min(imageToScaleCoordsToWidth.toFloat(), right)
                    val clampedBottom = min(imageToScaleCoordsToHeight.toFloat(), bottom)

                    if (clampedLeft < clampedRight && clampedTop < clampedBottom) {
                        candidates.add(
                            DetectionResult(
                                RectF(clampedLeft, clampedTop, clampedRight, clampedBottom),
                                labels.getOrElse(classId) { "unknown" },
                                confidence * maxClassScore
                            )
                        )
                    }
                }
            }
        }
        return nonMaxSuppression(candidates)
    }

    private fun nonMaxSuppression(candidates: List<DetectionResult>): List<DetectionResult> {
        if (candidates.isEmpty()) return emptyList()
        val sortedCandidates = candidates.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<DetectionResult>()
        val active = BooleanArray(sortedCandidates.size) { true }

        for (i in sortedCandidates.indices) {
            if (active[i]) {
                val a = sortedCandidates[i]
                selectedDetections.add(a)
                if (selectedDetections.size >= MAX_DETECTED_OBJECTS) break
                for (j in i + 1 until sortedCandidates.size) {
                    if (active[j]) {
                        val b = sortedCandidates[j]
                        if (calculateIoU(a.boundingBox, b.boundingBox) > IOU_THRESHOLD) {
                            active[j] = false
                        }
                    }
                }
            }
        }
        return selectedDetections
    }

    private fun calculateIoU(boxA: RectF, boxB: RectF): Float {
        val xA = max(boxA.left, boxB.left)
        val yA = max(boxA.top, boxB.top)
        val xB = min(boxA.right, boxB.right)
        val yB = min(boxA.bottom, boxB.bottom)
        val intersectionArea = max(0f, xB - xA) * max(0f, yB - yA)
        if (intersectionArea == 0f) return 0f
        val boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top)
        val boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top)
        return intersectionArea / (boxAArea + boxBArea - intersectionArea)
    }

    fun ImageProxy.toBitmap(): Bitmap? {
        if (this.planes.isEmpty()) {
            Log.e(TAG, "ImageProxy has no planes.")
            return null
        }
        return when (format) {
            ImageFormat.FLEX_RGBA_8888, ImageFormat.FLEX_RGB_888 -> {
                if (planes.isEmpty() || planes[0] == null) {
                    Log.e(TAG, "RGBA planes are missing or null.")
                    return null
                }
                val plane = planes[0]
                val buffer = plane.buffer.apply { rewind() }
                val pixelStride = plane.pixelStride
                val rowStride = plane.rowStride

                if (width <= 0 || height <= 0) {
                    Log.e(TAG, "Invalid image dimensions: $width x $height")
                    return null
                }
                if (pixelStride <= 0) {
                    Log.e(TAG, "Invalid pixelStride for RGBA: $pixelStride")
                    return null
                }

                val rowPadding = rowStride - pixelStride * width
                val bitmapBufferWidth = if (pixelStride > 0) rowStride / pixelStride else width


                if (bitmapBufferWidth <= 0) {
                    Log.e(TAG, "Invalid bitmapBufferWidth: $bitmapBufferWidth based on rowStride $rowStride and pixelStride $pixelStride")
                    return null
                }

                val tempBitmap: Bitmap
                try {
                    tempBitmap = Bitmap.createBitmap(bitmapBufferWidth, height, Bitmap.Config.ARGB_8888)
                    tempBitmap.copyPixelsFromBuffer(buffer)
                } catch (e: Exception) {
                    Log.e(TAG, "Error creating or copying to bitmap for RGBA: ${e.message}", e)
                    return null
                }


                if (rowPadding == 0 && bitmapBufferWidth == width) {
                    return tempBitmap
                } else if (bitmapBufferWidth >= width) {
                    val finalBitmap: Bitmap? = try {
                        Bitmap.createBitmap(tempBitmap, 0, 0, width, height)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error cropping RGBA bitmap: ${e.message}", e)
                        null
                    }
                    if (tempBitmap != finalBitmap && !tempBitmap.isRecycled) {
                        tempBitmap.recycle()
                    }
                    return finalBitmap
                } else {
                    Log.w(TAG, "Calculated bitmapBufferWidth $bitmapBufferWidth is less than image width $width. Returning tempBitmap possibly uncropped or incorrectly cropped.")
                    return tempBitmap
                }
            }
            ImageFormat.YUV_420_888 -> {
                val yBuffer = planes[0].buffer.apply { rewind() }
                val uBuffer = planes[1].buffer.apply { rewind() }
                val vBuffer = planes[2].buffer.apply { rewind() }

                val ySize = yBuffer.remaining()
                val uSize = uBuffer.remaining()
                val vSize = vBuffer.remaining()

                val nv21 = ByteArray(ySize + uSize + vSize)

                yBuffer.get(nv21, 0, ySize)
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + vSize, uSize)

                val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
                ByteArrayOutputStream().use { out ->
                    yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
                    val imageBytes = out.toByteArray()
                    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                }
            }
            else -> {
                Log.e(TAG, "Unsupported image format: $format for toBitmap conversion.")
                null
            }
        }
    }

    fun Bitmap.rotate(degrees: Float): Bitmap? {
        if (this.isRecycled) {
            Log.e(TAG, "Cannot rotate a recycled bitmap.")
            return null
        }
        return try {
            val matrix = Matrix().apply { postRotate(degrees) }
            Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OutOfMemoryError rotating bitmap", e)
            System.gc()
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error rotating bitmap", e)
            null
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    @Deprecated("Deprecated in Java")
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                if (processingEnabled && !isDestroyed && !isFinishing) {
                    loadCourtConfigurationAndStart()
                }
            } else {
                Toast.makeText(this, getString(R.string.permissions_not_granted_live), Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        Log.d(TAG, "onConfigurationChanged: New orientation: ${newConfig.orientation}")
        binding.root.requestApplyInsets()

        if (cameraProvider != null && allPermissionsGranted() && processingEnabled && !isDestroyed && !isFinishing) {
            binding.previewView.post {
                if (processingEnabled && !isDestroyed && !isFinishing) bindCameraUseCases()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy called")

        processingEnabled = false

        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "Error unbinding camera use cases in onDestroy: ${e.message}", e)
        }
        cameraProvider = null
        Log.d(TAG, "Camera use cases unbound.")

        cameraExecutor.shutdown()
        try {
            if (!cameraExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                Log.w(TAG, "Camera executor did not terminate in time. Forcing shutdown.")
                cameraExecutor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            Log.e(TAG, "Interrupted while waiting for camera executor to terminate.", e)
            cameraExecutor.shutdownNow()
            Thread.currentThread().interrupt()
        }
        Log.d(TAG, "CameraExecutor shutdown.")

        synchronized(interpreterLock) {
            tfliteInterpreter?.close()
            tfliteInterpreter = null
            Log.d(TAG, "TFLite interpreter closed.")
        }
    }

    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val RATIO_4_3_VALUE = 4.0 / 3.0
        private const val RATIO_16_9_VALUE = 16.0 / 9.0
    }
}