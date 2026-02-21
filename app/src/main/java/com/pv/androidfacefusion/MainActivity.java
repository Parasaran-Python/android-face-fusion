package com.pv.androidfacefusion;

import android.Manifest;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    
    private static final int REQUEST_PERMISSIONS = 100;
    
    private ImageView sourceImageView, targetImageView, resultImageView;
    private MaterialButton btnSelectSourceLocal, btnSelectSourceUrl;
    private MaterialButton btnSelectTargetLocal, btnSelectTargetUrl;
    private MaterialButton btnProcess, btnSaveResult;
    private LinearProgressIndicator progressBar;
    private MaterialCardView resultCard;
    
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
        
        btnSelectSourceLocal = findViewById(R.id.btnSelectSourceLocal);
        btnSelectSourceUrl = findViewById(R.id.btnSelectSourceUrl);
        btnSelectTargetLocal = findViewById(R.id.btnSelectTargetLocal);
        btnSelectTargetUrl = findViewById(R.id.btnSelectTargetUrl);
        btnProcess = findViewById(R.id.btnProcess);
        btnSaveResult = findViewById(R.id.btnSaveResult);
        
        progressBar = findViewById(R.id.progressBar);
        resultCard = findViewById(R.id.resultCard);
        
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
    
    private void initModels() {
        progressBar.setVisibility(View.VISIBLE);
        btnProcess.setEnabled(false);
        
        executorService.execute(() -> {
            try {
                // Check if models need to be downloaded
                ModelDownloader downloader = new ModelDownloader(this);
                
                if (!downloader.areAllModelsDownloaded()) {
                    runOnUiThread(() -> {
                        Toast.makeText(this, "Downloading AI models... This may take a few minutes on first launch", 
                            Toast.LENGTH_LONG).show();
                    });
                    
                    // Set up download callback
                    downloader.setCallback(new ModelDownloader.DownloadCallback() {
                        @Override
                        public void onProgress(String modelName, int progress) {
                            runOnUiThread(() -> {
                                // Update progress (could show in UI if desired)
                                Log.d("MainActivity", modelName + ": " + progress + "%");
                            });
                        }
                        
                        @Override
                        public void onComplete(String modelName) {
                            runOnUiThread(() -> {
                                Toast.makeText(MainActivity.this, 
                                    modelName + " downloaded!", Toast.LENGTH_SHORT).show();
                            });
                        }
                        
                        @Override
                        public void onError(String modelName, String error) {
                            Log.e("MainActivity", "Download error: " + modelName + " - " + error);
                        }
                    });
                }
                
                // Initialize all models (will download if needed)
                runOnUiThread(() -> {
                    Toast.makeText(this, "Loading face detection model...", Toast.LENGTH_SHORT).show();
                });
                faceDetector = new FaceDetector(this);
                faceDetector.initialize();
                
                runOnUiThread(() -> {
                    Toast.makeText(this, "Loading face recognition model...", Toast.LENGTH_SHORT).show();
                });
                faceEmbedder = new FaceEmbedder(this);
                faceEmbedder.initialize();
                
                runOnUiThread(() -> {
                    Toast.makeText(this, "Loading face swapping model...", Toast.LENGTH_SHORT).show();
                });
                faceSwapper = new FaceSwapper(this);
                faceSwapper.initialize();
                
                processor = new FaceFusionProcessor(faceDetector, faceEmbedder, faceSwapper);
                
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btnProcess.setEnabled(true);
                    long modelSize = downloader.getTotalModelSize();
                    Toast.makeText(this, "All models loaded successfully! (" + modelSize + " MB)", 
                        Toast.LENGTH_LONG).show();
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    showError("Failed to load models.\n\n" +
                        "Please check:\n" +
                        "1. Internet connection is active\n" +
                        "2. At least 800MB free storage\n" +
                        "3. Firewall allows connections to HuggingFace\n\n" +
                        "Models are downloaded automatically from:\n" +
                        "https://huggingface.co/leonelhs/insightface\n\n" +
                        "Error: " + e.getMessage());
                });
            }
        });
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
        
        btnSaveResult.setOnClickListener(v -> saveResult());
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
                
                // DEBUG: Log bitmap info
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
                
                // DEBUG: Log bitmap info
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
        
        progressBar.setVisibility(View.VISIBLE);
        btnProcess.setEnabled(false);
        resultCard.setVisibility(View.GONE);
        
        executorService.execute(() -> {
            try {
                Bitmap result = processor.processFaceFusion(sourceBitmap, targetBitmap);
                resultBitmap = result;
                
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btnProcess.setEnabled(true);
                    resultImageView.setImageBitmap(result);
                    resultCard.setVisibility(View.VISIBLE);
                    Toast.makeText(this, "Face swap completed!", Toast.LENGTH_SHORT).show();
                });
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
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
        
        // Check permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            permissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            return;
        }
        
        executorService.execute(() -> {
            try {
                String fileName = "face_fusion_" + System.currentTimeMillis() + ".jpg";
                
                // Save using MediaStore (Android 10+)
                ContentValues values = new ContentValues();
                values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
                values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);
                
                Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                if (uri != null) {
                    FileOutputStream out = (FileOutputStream) getContentResolver().openOutputStream(uri);
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