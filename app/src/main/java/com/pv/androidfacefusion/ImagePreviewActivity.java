package com.pv.androidfacefusion;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

public class ImagePreviewActivity extends AppCompatActivity {

    // Static bitmap holder to avoid Intent parcel size limits.
    // Caller sets this before launching, activity reads and nulls it.
    private static Bitmap pendingBitmap;

    public static void setPendingBitmap(Bitmap bitmap) {
        pendingBitmap = bitmap;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Full-screen immersive
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowInsetsControllerCompat controller =
                WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView());
        controller.hide(WindowInsetsCompat.Type.systemBars());
        controller.setSystemBarsBehavior(
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);

        setContentView(R.layout.activity_image_preview);

        ZoomableImageView imageView = findViewById(R.id.previewImageView);
        findViewById(R.id.btnClose).setOnClickListener(v -> finish());

        if (pendingBitmap != null) {
            imageView.setImageBitmap(pendingBitmap);
            pendingBitmap = null; // Release the static reference
        } else {
            finish();
        }
    }
}
