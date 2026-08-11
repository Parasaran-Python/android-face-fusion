package com.pv.androidfacefusion;

import android.Manifest;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.bumptech.glide.Glide;
import com.bumptech.glide.request.FutureTarget;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import android.widget.HorizontalScrollView;
import com.google.android.material.chip.Chip;
import com.google.android.material.chip.ChipGroup;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_PERMISSIONS = 100;

    private ImageView sourceImageView, resultImageView;
    private FaceOverlayImageView targetImageView;
    private HorizontalScrollView targetFaceChipScrollView;
    private ChipGroup targetFaceChipGroup;
    private TextView targetFaceStatusText;
    private List<FaceDetector.Face> targetFacesList = new ArrayList<>();

    private MaterialButton btnSelectSourceLocal, btnSelectSourceUrl;
    private MaterialButton btnSelectTargetLocal, btnSelectTargetUrl;
    private MaterialButton btnProcess, btnSaveResult, btnShareResult, btnReset;
    private LinearProgressIndicator progressBar;
    private MaterialCardView resultCard;

    // Download overlay views
    private FrameLayout downloadOverlay;
    private TextView overlayTitle, overlayStatus, overlayPercent;
    private LinearProgressIndicator overlayProgress;

    private Bitmap sourceBitmap, targetBitmap, resultBitmap;

    private FaceDetector faceDetector;
    private FaceEmbedder faceEmbedder;
    private FaceSwapper faceSwapper;
    private FaceFusionProcessor processor;

    private ExecutorService executorService;

    private boolean isSelectingSource = true;

    private ActivityResultLauncher<Intent> imagePickerLauncher;
    private ActivityResultLauncher<String> permissionLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        initViews();
        initModels();
        setupListeners();
        requestPermissions();
    }

    private void initViews() {
        sourceImageView = findViewById(R.id.sourceImageView);
        targetImageView = findViewById(R.id.targetImageView);
        resultImageView = findViewById(R.id.resultImageView);

        targetFaceChipScrollView = findViewById(R.id.targetFaceChipScrollView);
        targetFaceChipGroup = findViewById(R.id.targetFaceChipGroup);
        targetFaceStatusText = findViewById(R.id.targetFaceStatusText);

        btnSelectSourceLocal = findViewById(R.id.btnSelectSourceLocal);
        btnSelectSourceUrl = findViewById(R.id.btnSelectSourceUrl);
        btnSelectTargetLocal = findViewById(R.id.btnSelectTargetLocal);
        btnSelectTargetUrl = findViewById(R.id.btnSelectTargetUrl);
        btnProcess = findViewById(R.id.btnProcess);
        btnReset = findViewById(R.id.btnReset);
        btnSaveResult = findViewById(R.id.btnSaveResult);
        btnShareResult = findViewById(R.id.btnShareResult);

        progressBar = findViewById(R.id.progressBar);
        resultCard = findViewById(R.id.resultCard);

        // Download overlay
        downloadOverlay = findViewById(R.id.downloadOverlay);
        overlayTitle = findViewById(R.id.overlayTitle);
        overlayStatus = findViewById(R.id.overlayStatus);
        overlayPercent = findViewById(R.id.overlayPercent);
        overlayProgress = findViewById(R.id.overlayProgress);

        executorService = Executors.newSingleThreadExecutor();

        // Setup image picker launcher
        imagePickerLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    if (imageUri != null) {
                        loadImageFromUri(imageUri, isSelectingSource);
                    }
                }
            }
        );

        // Setup permission launcher
        permissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestPermission(),
            isGranted -> {
                if (!isGranted) {
                    Toast.makeText(this, "Storage permission is required to save images",
                        Toast.LENGTH_SHORT).show();
                }
            }
        );
    }

    private void showOverlay(String title, String status) {
        downloadOverlay.setVisibility(View.VISIBLE);
        overlayTitle.setText(title);
        overlayStatus.setText(status);
        overlayPercent.setText("");
        overlayProgress.setIndeterminate(true);
    }

    private void updateOverlay(String status, int progress) {
        overlayStatus.setText(status);
        if (progress >= 0) {
            overlayProgress.setIndeterminate(false);
            overlayProgress.setProgressCompat(progress, true);
            overlayPercent.setText(progress + "%");
        } else {
            overlayProgress.setIndeterminate(true);
            overlayPercent.setText("");
        }
    }

    private void hideOverlay() {
        downloadOverlay.setVisibility(View.GONE);
    }

    private void initModels() {
        btnProcess.setEnabled(false);

        executorService.execute(() -> {
            try {
                ModelDownloader downloader = new ModelDownloader(this);
                boolean needsDownload = !downloader.areAllModelsDownloaded();

                runOnUiThread(() -> {
                    if (needsDownload) {
                        showOverlay("Setting up AI Models", "Preparing download...");
                    } else {
                        showOverlay("Loading AI Models", "Initializing...");
                    }
                });

                if (needsDownload) {
                    downloader.setCallback(new ModelDownloader.DownloadCallback() {
                        @Override
                        public void onProgress(String modelName, int progress) {
                            String displayName = getModelDisplayName(modelName);
                            runOnUiThread(() -> updateOverlay("Downloading " + displayName, progress));
                        }

                        @Override
                        public void onComplete(String modelName) {
                            String displayName = getModelDisplayName(modelName);
                            runOnUiThread(() -> updateOverlay(displayName + " downloaded", 100));
                        }

                        @Override
                        public void onError(String modelName, String error) {
                            Log.e("MainActivity", "Download error: " + modelName + " - " + error);
                        }
                    });
                }

                // Load detector
                runOnUiThread(() -> updateOverlay(needsDownload
                        ? "Downloading Face Detector (~16 MB)"
                        : "Loading Face Detector", -1));
                faceDetector = new FaceDetector(this);
                faceDetector.initialize();

                // Load embedder
                runOnUiThread(() -> updateOverlay(needsDownload
                        ? "Downloading Face Embedder (~166 MB)"
                        : "Loading Face Embedder", -1));
                faceEmbedder = new FaceEmbedder(this);
                faceEmbedder.initialize();

                // Load swapper
                runOnUiThread(() -> updateOverlay(needsDownload
                        ? "Downloading Face Swapper (~553 MB)"
                        : "Loading Face Swapper", -1));
                faceSwapper = new FaceSwapper(this);
                faceSwapper.initialize();

                processor = new FaceFusionProcessor(faceDetector, faceEmbedder, faceSwapper);

                runOnUiThread(() -> {
                    hideOverlay();
                    btnProcess.setEnabled(true);
                    Toast.makeText(this, "All models loaded!", Toast.LENGTH_SHORT).show();
                    if (targetBitmap != null) {
                        detectTargetFacesAsync(targetBitmap);
                    }
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> {
                    hideOverlay();
                    showError("Failed to load models.\n\n" +
                        "Please check:\n" +
                        "1. Internet connection is active\n" +
                        "2. At least 800MB free storage\n" +
                        "3. Firewall allows connections to HuggingFace\n\n" +
                        "Error: " + e.getMessage());
                });
            }
        });
    }

    private String getModelDisplayName(String modelName) {
        switch (modelName) {
            case "det_10g.onnx": return "Face Detector";
            case "w600k_r50.onnx": return "Face Embedder";
            case "inswapper_128.onnx": return "Face Swapper";
            default: return modelName;
        }
    }

    private void setupListeners() {
        btnSelectSourceLocal.setOnClickListener(v -> {
            isSelectingSource = true;
            openImagePicker();
        });

        btnSelectSourceUrl.setOnClickListener(v -> {
            isSelectingSource = true;
            showUrlInputDialog();
        });

        btnSelectTargetLocal.setOnClickListener(v -> {
            isSelectingSource = false;
            openImagePicker();
        });

        btnSelectTargetUrl.setOnClickListener(v -> {
            isSelectingSource = false;
            showUrlInputDialog();
        });

        btnProcess.setOnClickListener(v -> processFaceFusion());

        btnReset.setOnClickListener(v -> resetSession());

        btnSaveResult.setOnClickListener(v -> saveResult());
        btnShareResult.setOnClickListener(v -> shareResult());

        // Image preview on tap
        sourceImageView.setOnClickListener(v -> openImagePreview(sourceBitmap));
        resultImageView.setOnClickListener(v -> openImagePreview(resultBitmap));

        // Sync canvas tap with face selection chips
        targetImageView.setOnFaceSelectedListener(selectedIndices -> syncChipsWithOverlay());
    }

    private void openImagePreview(Bitmap bitmap) {
        if (bitmap == null) return;
        ImagePreviewActivity.setPendingBitmap(bitmap);
        startActivity(new Intent(this, ImagePreviewActivity.class));
    }

    private void resetSession() {
        if (sourceBitmap != null) { sourceBitmap.recycle(); sourceBitmap = null; }
        if (targetBitmap != null) { targetBitmap.recycle(); targetBitmap = null; }
        if (resultBitmap != null) { resultBitmap.recycle(); resultBitmap = null; }

        sourceImageView.setImageDrawable(null);
        targetImageView.setImageDrawable(null);
        targetImageView.setFaces(null);
        targetFacesList.clear();

        targetFaceChipGroup.removeAllViews();
        targetFaceStatusText.setVisibility(View.GONE);
        targetFaceChipScrollView.setVisibility(View.GONE);

        resultImageView.setImageDrawable(null);
        resultCard.setVisibility(View.GONE);
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        imagePickerLauncher.launch(intent);
    }

    private void showUrlInputDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter Image URL");

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        input.setHint("https://example.com/image.jpg");
        builder.setView(input);

        builder.setPositiveButton("Load", (dialog, which) -> {
            String url = input.getText().toString().trim();
            if (!url.isEmpty()) {
                loadImageFromUrl(url, isSelectingSource);
            } else {
                Toast.makeText(this, "Please enter a valid URL", Toast.LENGTH_SHORT).show();
            }
        });

        builder.setNegativeButton("Cancel", (dialog, which) -> dialog.cancel());
        builder.show();
    }

    private void loadImageFromUri(Uri uri, boolean isSource) {
        executorService.execute(() -> {
            try {
                InputStream inputStream = getContentResolver().openInputStream(uri);
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                inputStream.close();

                Log.d("MainActivity", "Loaded " + (isSource ? "SOURCE" : "TARGET") + " image:");
                Log.d("MainActivity", "  Size: " + bitmap.getWidth() + "x" + bitmap.getHeight());
                Log.d("MainActivity", "  Config: " + bitmap.getConfig());

                // Ensure ARGB_8888 format
                if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
                    Log.w("MainActivity", "  Converting to ARGB_8888 from " + bitmap.getConfig());
                    Bitmap converted = bitmap.copy(Bitmap.Config.ARGB_8888, false);
                    bitmap.recycle();
                    bitmap = converted;
                }

                // Resize if too large
                if (bitmap.getWidth() > 1024 || bitmap.getHeight() > 1024) {
                    Log.d("MainActivity", "  Resizing from " + bitmap.getWidth() + "x" + bitmap.getHeight());
                    bitmap = ImageUtils.resizeImage(bitmap, 1024);
                    Log.d("MainActivity", "  Resized to " + bitmap.getWidth() + "x" + bitmap.getHeight());
                }

                Bitmap finalBitmap = bitmap;
                runOnUiThread(() -> {
                    if (isSource) {
                        sourceBitmap = finalBitmap;
                        sourceImageView.setImageBitmap(finalBitmap);
                    } else {
                        targetBitmap = finalBitmap;
                        targetImageView.setImageBitmap(finalBitmap);
                        detectTargetFacesAsync(finalBitmap);
                    }
                });
            } catch (IOException e) {
                e.printStackTrace();
                runOnUiThread(() -> showError("Failed to load image: " + e.getMessage()));
            }
        });
    }

    private void loadImageFromUrl(String url, boolean isSource) {
        progressBar.setVisibility(View.VISIBLE);

        executorService.execute(() -> {
            try {
                FutureTarget<Bitmap> futureTarget = Glide.with(this)
                    .asBitmap()
                    .load(url)
                    .submit();

                Bitmap bitmap = futureTarget.get();

                Log.d("MainActivity", "Loaded " + (isSource ? "SOURCE" : "TARGET") + " from URL:");
                Log.d("MainActivity", "  Size: " + bitmap.getWidth() + "x" + bitmap.getHeight());
                Log.d("MainActivity", "  Config: " + bitmap.getConfig());

                // Ensure ARGB_8888 format
                if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
                    Log.w("MainActivity", "  Converting to ARGB_8888 from " + bitmap.getConfig());
                    Bitmap converted = bitmap.copy(Bitmap.Config.ARGB_8888, false);
                    bitmap.recycle();
                    bitmap = converted;
                }

                // Resize if too large
                if (bitmap.getWidth() > 1024 || bitmap.getHeight() > 1024) {
                    bitmap = ImageUtils.resizeImage(bitmap, 1024);
                }

                Bitmap finalBitmap = bitmap;
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    if (isSource) {
                        sourceBitmap = finalBitmap;
                        sourceImageView.setImageBitmap(finalBitmap);
                    } else {
                        targetBitmap = finalBitmap;
                        targetImageView.setImageBitmap(finalBitmap);
                        detectTargetFacesAsync(finalBitmap);
                    }
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    showError("Failed to load image from URL: " + e.getMessage());
                });
            }
        });
    }

    private void detectTargetFacesAsync(Bitmap bitmap) {
        if (faceDetector == null || bitmap == null) return;
        executorService.execute(() -> {
            try {
                List<FaceDetector.Face> faces = faceDetector.detectFaces(bitmap);
                runOnUiThread(() -> updateTargetFaceSelectionUI(faces));
            } catch (Exception e) {
                Log.e("MainActivity", "Error detecting target faces", e);
                runOnUiThread(() -> {
                    if (targetFaceStatusText != null) targetFaceStatusText.setVisibility(View.GONE);
                    if (targetFaceChipScrollView != null) targetFaceChipScrollView.setVisibility(View.GONE);
                });
            }
        });
    }

    private boolean isSyncingChips = false;

    private void updateTargetFaceSelectionUI(List<FaceDetector.Face> faces) {
        targetFacesList = (faces != null) ? faces : new ArrayList<>();
        targetImageView.setFaces(targetFacesList);

        targetFaceChipGroup.removeAllViews();
        targetFaceChipGroup.setOnCheckedStateChangeListener(null);

        if (targetFacesList.isEmpty()) {
            targetFaceStatusText.setText("No face detected in target image");
            targetFaceStatusText.setVisibility(View.VISIBLE);
            targetFaceChipScrollView.setVisibility(View.GONE);
            return;
        }

        targetFaceStatusText.setVisibility(View.VISIBLE);

        if (targetFacesList.size() == 1) {
            targetFaceStatusText.setText("1 face detected in target image:");
            targetFaceChipScrollView.setVisibility(View.VISIBLE);

            Chip chip = createFaceChip("👤 Face 1", 0);
            chip.setChecked(true);
            targetFaceChipGroup.addView(chip);
        } else {
            targetFaceStatusText.setText(targetFacesList.size() + " faces detected. Tap faces on image or chips below to toggle:");
            targetFaceChipScrollView.setVisibility(View.VISIBLE);

            // Add "Select All" master chip (index -1)
            Chip allChip = createFaceChip("✨ Select All (" + targetFacesList.size() + ")", -1);
            allChip.setChecked(true);
            targetFaceChipGroup.addView(allChip);

            // Add individual face chips
            for (int i = 0; i < targetFacesList.size(); i++) {
                Chip chip = createFaceChip("👤 Face " + (i + 1), i);
                chip.setChecked(true);
                targetFaceChipGroup.addView(chip);
            }
        }

        // Attach click listeners for 2-way master toggle and individual chip sync
        for (int i = 0; i < targetFaceChipGroup.getChildCount(); i++) {
            View child = targetFaceChipGroup.getChildAt(i);
            if (child instanceof Chip && child.getTag() instanceof Integer) {
                Chip chip = (Chip) child;
                int index = (Integer) chip.getTag();
                chip.setOnClickListener(v -> {
                    if (isSyncingChips) return;
                    if (index == -1) {
                        // Master toggle: if Select All is checked -> select all faces; if unchecked -> clear selection
                        if (chip.isChecked()) {
                            targetImageView.setSelectedFaceIndex(-1);
                        } else {
                            targetImageView.setSelectedFaceIndices(new HashSet<>());
                        }
                    } else {
                        // Individual face toggle
                        targetImageView.toggleFaceIndex(index);
                    }
                    syncChipsWithOverlay();
                });
            }
        }
    }

    private Chip createFaceChip(String label, int index) {
        Chip chip = new Chip(this);
        chip.setText(label);
        chip.setCheckable(true);
        chip.setClickable(true);
        chip.setTag(index);
        chip.setId(View.generateViewId());
        return chip;
    }

    private void selectChipForFaceIndex(int index) {
        syncChipsWithOverlay();
    }

    private void syncChipsWithOverlay() {
        if (targetFaceChipGroup == null || targetFacesList == null) return;
        isSyncingChips = true;
        Set<Integer> selected = targetImageView.getSelectedFaceIndices();
        boolean allSelected = (selected.size() == targetFacesList.size() && !targetFacesList.isEmpty());

        for (int i = 0; i < targetFaceChipGroup.getChildCount(); i++) {
            View child = targetFaceChipGroup.getChildAt(i);
            if (child instanceof Chip && child.getTag() instanceof Integer) {
                Chip chip = (Chip) child;
                int tag = (Integer) chip.getTag();
                if (tag == -1) {
                    // Auto-select "Select All" when all individual faces are selected
                    chip.setChecked(allSelected);
                } else {
                    // Auto-select individual chip if face is in selected set
                    chip.setChecked(selected.contains(tag));
                }
            }
        }
        isSyncingChips = false;
    }

    private void processFaceFusion() {
        if (sourceBitmap == null || targetBitmap == null) {
            Toast.makeText(this, "Please select both source and target images",
                Toast.LENGTH_SHORT).show();
            return;
        }

        if (processor == null) {
            Toast.makeText(this, "Models are still loading, please wait",
                Toast.LENGTH_SHORT).show();
            return;
        }

        final Set<Integer> selectedFaceIndices = targetImageView.getSelectedFaceIndices();
        if (selectedFaceIndices == null || selectedFaceIndices.isEmpty()) {
            Toast.makeText(this, "Please select at least one target face to swap",
                Toast.LENGTH_SHORT).show();
            return;
        }

        btnProcess.setEnabled(false);
        resultCard.setVisibility(View.GONE);
        showOverlay("Swapping Faces", "Processing face swap...");

        executorService.execute(() -> {
            try {
                runOnUiThread(() -> updateOverlay("Processing face swap...", -1));
                Bitmap result = processor.processFaceFusion(sourceBitmap, targetBitmap, selectedFaceIndices);
                resultBitmap = result;

                runOnUiThread(() -> {
                    hideOverlay();
                    btnProcess.setEnabled(true);
                    resultImageView.setImageBitmap(result);
                    resultCard.setVisibility(View.VISIBLE);
                    Toast.makeText(this, "Face swap completed!", Toast.LENGTH_SHORT).show();
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> {
                    hideOverlay();
                    btnProcess.setEnabled(true);
                    showError("Face fusion failed: " + e.getMessage());
                });
            }
        });
    }

    private void saveResult() {
        if (resultBitmap == null) {
            Toast.makeText(this, "No result to save", Toast.LENGTH_SHORT).show();
            return;
        }

        // WRITE_EXTERNAL_STORAGE is only needed on Android 9 (API 28) and below.
        // On Android 10+ (API 29+), MediaStore API works without this permission.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
                permissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE);
                return;
            }
        }

        executorService.execute(() -> {
            try {
                String fileName = "face_fusion_" + System.currentTimeMillis() + ".jpg";

                ContentValues values = new ContentValues();
                values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
                values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);

                Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                if (uri != null) {
                    OutputStream out = getContentResolver().openOutputStream(uri);
                    resultBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out);
                    out.close();

                    runOnUiThread(() ->
                        Toast.makeText(this, "Image saved to Pictures folder", Toast.LENGTH_LONG).show()
                    );
                }
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> showError("Failed to save image: " + e.getMessage()));
            }
        });
    }

    private void shareResult() {
        if (resultBitmap == null) {
            Toast.makeText(this, "No result to share", Toast.LENGTH_SHORT).show();
            return;
        }

        executorService.execute(() -> {
            try {
                // Save bitmap to a temp file in cache for sharing
                File shareDir = new File(getCacheDir(), "shared_images");
                if (!shareDir.exists()) shareDir.mkdirs();
                File shareFile = new File(shareDir, "face_fusion_share.jpg");

                FileOutputStream out = new FileOutputStream(shareFile);
                resultBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out);
                out.close();

                Uri contentUri = FileProvider.getUriForFile(
                    this, getPackageName() + ".fileprovider", shareFile);

                runOnUiThread(() -> {
                    Intent shareIntent = new Intent(Intent.ACTION_SEND);
                    shareIntent.setType("image/jpeg");
                    shareIntent.putExtra(Intent.EXTRA_STREAM, contentUri);
                    shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    startActivity(Intent.createChooser(shareIntent, "Share Face Fusion Result"));
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> showError("Failed to share image: " + e.getMessage()));
            }
        });
    }

    private void requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.INTERNET)
            != PackageManager.PERMISSION_GRANTED) {
            permissionLauncher.launch(Manifest.permission.INTERNET);
        }
    }

    private void showError(String message) {
        new AlertDialog.Builder(this)
            .setTitle("Error")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executorService.shutdown();

        if (faceDetector != null) faceDetector.close();
        if (faceEmbedder != null) faceEmbedder.close();
        if (faceSwapper != null) faceSwapper.close();

        if (sourceBitmap != null) sourceBitmap.recycle();
        if (targetBitmap != null) targetBitmap.recycle();
        if (resultBitmap != null) resultBitmap.recycle();
    }
}
