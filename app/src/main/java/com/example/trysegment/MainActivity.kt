package com.example.trysegment

//package org.pytorch.imagesegmentation
import android.app.Activity
import android.content.Context
import android.content.ContentValues

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.icu.text.SimpleDateFormat
import android.graphics.Canvas
import android.graphics.Paint
import android.Manifest
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale




class MainActivity : AppCompatActivity(), Runnable {
    private lateinit var viewFinder: PreviewView
    private lateinit var mImageView: ImageView
    private lateinit var mButtonSegment: Button
    private lateinit var mProgressBar: ProgressBar
    private lateinit var outputDirectory: File
    private lateinit var btnDiscard: Button
    private lateinit var btnTakePhoto: Button
    private lateinit var btnLoadImage: Button
    private lateinit var btnSaveMask: Button
    private lateinit var alphaSeekBar: SeekBar
    private lateinit var btnToggleFlash: ImageButton // Change to ImageButton

    private var originalBitmap: Bitmap? = null
    private var segmentationMask: Bitmap? = null
    private var mBitmap: Bitmap? = null
    private var mModule: Module? = null
    private var mImagename = "deeplab.jpg"
    private val TAG = "MainActivity"
    private var imageCapture: ImageCapture? = null

    private var isFlashEnabled = true

    companion object {
        private const val IMAGE_PICK_CODE = 1000
        private const val CLASSNUM = 7
        private const val CRACK = 6
        private const val SPALLING = 5
        private const val CORROSION = 4
        private const val EFFLORESCENCE = 3
        private const val VEGETATION = 2
        private const val CONTROL_POINT = 1


        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }

            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            return file.absolutePath
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        mImageView = findViewById(R.id.imageView)
        btnDiscard = findViewById(R.id.btnDiscard)  // Initialize btnDiscard
        btnLoadImage = findViewById(R.id.btnLoadImage)
        mButtonSegment = findViewById(R.id.btnSegment)
        mProgressBar = findViewById(R.id.progressBar)
        outputDirectory = getOutputDirectory()
        viewFinder = findViewById(R.id.viewFinder)
        btnTakePhoto = findViewById(R.id.btnTakePhoto)
        btnToggleFlash = findViewById(R.id.btnToggleFlash)
        alphaSeekBar = findViewById(R.id.alphaSeekBar)
        btnSaveMask = findViewById(R.id.btnSaveMask)

        btnToggleFlash.setOnClickListener {
            isFlashEnabled = !isFlashEnabled
            // Update the button text or icon based on the flash state
            btnToggleFlash.setImageResource(if (isFlashEnabled) R.drawable.ic_flash_on else R.drawable.ic_flash_off)
            // Reconfigure the camera with the new flash setting
            startCamera()
        }

        btnDiscard.setOnClickListener {
            mImageView.visibility = View.GONE
            mImageView.setImageURI(null)
            viewFinder.visibility = View.VISIBLE
            btnTakePhoto.visibility = View.VISIBLE
            btnDiscard.visibility = View.GONE
            btnSaveMask.visibility = View.GONE
            btnLoadImage.visibility = View.VISIBLE
            mButtonSegment.visibility = View.VISIBLE
            mImageView = findViewById(R.id.imageView)
            //mImageView.setImageBitmap(mBitmap)
            mImageView.visibility = View.GONE // Make the ImageView invisible at app start
            startCamera()
        }

        btnLoadImage.setOnClickListener {
            openImagePicker()
        }

        mButtonSegment.setOnClickListener {
            mButtonSegment.isEnabled = false
            mProgressBar.visibility = ProgressBar.VISIBLE
            mButtonSegment.text = getString(R.string.run_model)

            Thread(this).start()

            mButtonSegment.visibility = View.GONE
            alphaSeekBar.visibility = View.VISIBLE
            btnLoadImage.visibility = View.GONE
            btnTakePhoto.visibility = View.GONE

            // Show "Save Mask" and "Discard" buttons
            btnSaveMask.visibility = View.VISIBLE
        }

        btnTakePhoto.setOnClickListener {
            takePhoto()
        }

        alphaSeekBar.visibility = View.GONE
        alphaSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                Log.d(TAG, "SeekBar Progress: $progress")
                adjustOverlayTransparency(progress)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {
            }
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
            }
        })

        btnSaveMask.setOnClickListener {
            saveMaskToStorage()
        }

        startCamera()

        try {
            mModule = LiteModuleLoader.load(
                assetFilePath(
                    applicationContext,
                    "deeplabv3_scripted_optimized.ptl"
                )
            )
        } catch (e: IOException) {
            Log.e("ImageSegmentation", "Error reading assets", e)
            finish()
        }
    }

    private fun startCamera() {
        btnToggleFlash.visibility = View.VISIBLE
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setFlashMode(if (isFlashEnabled) ImageCapture.FLASH_MODE_ON else ImageCapture.FLASH_MODE_OFF)
                .build()

            try {
                cameraProvider.unbindAll() // Unbind any use cases before rebinding
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture
                )
            } catch (exc: Exception) {
                Log.e("MainActivity", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        btnToggleFlash.visibility = View.GONE

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll() // Unbind all use cases and stop the cam
        }, ContextCompat.getMainExecutor(this))
    }

    private fun showCameraFeed() {
        runOnUiThread {
            mImageView.visibility = View.GONE // Hide the ImageView
            viewFinder.visibility = View.VISIBLE // Show the camera feed
        }
    }

    private fun takePhoto() {
        val photoFile = File(
            outputDirectory,
            SimpleDateFormat(
                Constants.FILE_NAME_FORMAT,
                Locale.getDefault()
            ).format(System.currentTimeMillis()) + ".jpg"
        )
        val storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)

        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        imageCapture?.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    Log.d("MainActivity", "Photo capture succeeded: $savedUri")

                    // Load the bitmap from the URI and downscale it for inference
                    val bitmap = loadBitmapFromUri(savedUri)?.let { originalBitmap ->
                        downscaleBitmapKeepingAspectRatio(originalBitmap, 600, 800)
                    }

                    Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE).also { mediaScanIntent ->
                        mediaScanIntent.data = savedUri
                        sendBroadcast(mediaScanIntent)
                    }

                    runOnUiThread {
                        if (bitmap != null) {
                            mImageView.visibility = View.VISIBLE
                            mImageView.setImageBitmap(bitmap)
                            mBitmap = bitmap // Set the downscaled bitmap for further processing
                            // Hide the PreviewView to stop showing the camera feed
                            viewFinder.visibility = View.INVISIBLE
                            btnTakePhoto.visibility = View.GONE
                            btnDiscard.visibility = View.VISIBLE
                            stopCamera()
                        } else {
                            Log.e("MainActivity", "Failed to load and downscale bitmap from $savedUri")
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("MainActivity", "Photo capture failed: ${exception.message}", exception)
                }
            })
    }

    fun adjustOverlayTransparency(alphaValue: Int) {
        val alpha = (alphaValue * 255) / 100 // Convert progress to alpha
        Log.d(TAG, "Adjusting overlay transparency to alpha: $alpha")

        mBitmap?.let { original ->
            segmentationMask?.let { mask ->
                // Create a new bitmap to store the result
                val combinedBitmap = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)

                // Canvas to combine the original bitmap and the overlay
                val canvas = Canvas(combinedBitmap)
                canvas.drawBitmap(original, 0f, 0f, null) // Draw original bitmap

                // Paint to apply alpha to the overlay
                val paint = Paint().apply { this.alpha = alpha }
                canvas.drawBitmap(mask, 0f, 0f, paint) // Draw overlay with alpha

                // Update the ImageView with the combined bitmap
                runOnUiThread { mImageView.setImageBitmap(combinedBitmap) }
            } ?: Log.d(TAG, "Segmentation mask bitmap is null")
        } ?: Log.d(TAG, "Original bitmap is null")
    }

    private fun openImagePicker() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, IMAGE_PICK_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == IMAGE_PICK_CODE && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                val bitmap = loadBitmapFromUri(uri)
                bitmap?.let {
                    processBitmap(it) // Process and display the bitmap
                }
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT < 28) {
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            } else {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source)
            }.copy(Bitmap.Config.ARGB_8888, true) // Convert to a mutable bitmap
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load bitmap from URI", e)
            null
        }
    }

    // Call this function after you've successfully loaded a bitmap using loadBitmapFromUri
    private fun processBitmap(bitmap: Bitmap) {
        val downscaledBitmap = downscaleBitmapKeepingAspectRatio(bitmap, 600, 800)
        runOnUiThread {
            mImageView.setImageBitmap(downscaledBitmap)
            mImageView.visibility = View.VISIBLE // Make the ImageView visible
            viewFinder.visibility = View.GONE // Hide the camera feed
            btnTakePhoto.visibility = View.GONE
            btnDiscard.visibility = View.VISIBLE // Show the Discard button
        }
        mBitmap =
            downscaledBitmap // Update mBitmap with the downscaled version for further processing
    }

    private fun downscaleBitmapKeepingAspectRatio(
        sourceBitmap: Bitmap,
        maxWidth: Int,
        maxHeight: Int
    ): Bitmap {
        val originalWidth = sourceBitmap.width
        val originalHeight = sourceBitmap.height

        // Compute the scaling factors to maintain aspect ratio
        val widthScale = maxWidth.toFloat() / originalWidth
        val heightScale = maxHeight.toFloat() / originalHeight
        val scale = minOf(widthScale, heightScale)

        // Calculate the scaled dimensions
        val scaledWidth = (originalWidth * scale).toInt()
        val scaledHeight = (originalHeight * scale).toInt()

        // Create and return the scaled bitmap
        return Bitmap.createScaledBitmap(sourceBitmap, scaledWidth, scaledHeight, true)
    }

    private fun saveMaskToStorage() {
        segmentationMask?.let { bitmap -> // Ensure this is your segmentation mask bitmap
            val resolver = contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, "Mask_${System.currentTimeMillis()}.png") // Name of the file
                put(MediaStore.MediaColumns.MIME_TYPE, "image/png") // Type of the file
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES) // Save directory
            }

            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            try {
                uri?.let {
                    resolver.openOutputStream(it)?.use { out ->
                        // Now 'out' is safely cast to 'OutputStream' and cannot be null inside this block
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) // Use 'bitmap' which refers to 'transferredBitmap'
                        Toast.makeText(this, "Saved segmentation mask.", Toast.LENGTH_SHORT).show()
                    } ?: runOnUiThread {
                        Toast.makeText(this, "Failed to open output stream.", Toast.LENGTH_SHORT).show()
                    }
                } ?: runOnUiThread {
                    Toast.makeText(this, "Failed to create new MediaStore record.", Toast.LENGTH_SHORT).show()
                }
            } catch (e: IOException) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this, "Failed to save mask", Toast.LENGTH_SHORT).show()
                }
            }
        } ?: runOnUiThread {
            Toast.makeText(this, "Segmentation mask bitmap is null.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun run() {
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            mBitmap!!,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        val startTime = SystemClock.elapsedRealtime()
        val outTensors = mModule!!.forward(IValue.from(inputTensor)).toDictStringKey()
        val inferenceTime = SystemClock.elapsedRealtime() - startTime
        Log.d("ImageSegmentation", "inference time (ms): $inferenceTime")

        val outputTensor = outTensors["out"]!!.toTensor()
        val scores = outputTensor.dataAsFloatArray
        val width = mBitmap!!.width
        val height = mBitmap!!.height
        val intValues = IntArray(width * height)
        Log.d("ImageSegmentation", "Output tensor shape: ${outputTensor.shape().contentToString()}")

        for (j in 0 until height) {
            for (k in 0 until width) {
                var maxIndex = 0
                var maxScore = -Float.MAX_VALUE
                for (i in 0 until CLASSNUM) {
                    val score = scores[i * (width * height) + j * width + k]
                    if (score > maxScore) {
                        maxScore = score
                        maxIndex = i
                    }
                }
                intValues[j * width + k] = when (maxIndex) {
                    CRACK -> 0xFFFFFFFF.toInt() // White
                    SPALLING -> 0xFFFF0000.toInt() // Red
                    CORROSION -> 0xFFFFFF00.toInt() // Yellow
                    EFFLORESCENCE -> 0xFF00FFFF.toInt() // Cyan
                    VEGETATION -> 0xFF00FF00.toInt() // Green
                    CONTROL_POINT -> 0xFF0000FF.toInt() // Blue
                    else -> 0xFF000000.toInt()
                }
            }
        }

        val bmpSegmentation = Bitmap.createScaledBitmap(mBitmap!!, width, height, true)
        val outputBitmap = bmpSegmentation.copy(Bitmap.Config.ARGB_8888, true)
        outputBitmap.setPixels(intValues, 0, width, 0, 0, width, height)
        val transferredBitmap =
            Bitmap.createScaledBitmap(outputBitmap, mBitmap!!.width, mBitmap!!.height, true)

        runOnUiThread {
            segmentationMask = transferredBitmap
            mImageView.setImageBitmap(transferredBitmap)
            mButtonSegment.isEnabled = true
            mButtonSegment.text = getString(R.string.segment)
            mProgressBar.visibility = ProgressBar.INVISIBLE
        }
    }

    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() }
        }
        return if (mediaDir != null && mediaDir.exists()) mediaDir else filesDir
    }
}