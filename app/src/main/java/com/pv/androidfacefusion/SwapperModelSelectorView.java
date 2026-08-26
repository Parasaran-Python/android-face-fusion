package com.pv.androidfacefusion;

import android.app.AlertDialog;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.AttributeSet;
import android.widget.Toast;

import com.google.android.material.button.MaterialButton;

/** Compact quality selector for swapper model plus identity strength. */
public class SwapperModelSelectorView extends MaterialButton {
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private volatile boolean downloading;

    public SwapperModelSelectorView(Context context) {
        super(context);
        init();
    }

    public SwapperModelSelectorView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public SwapperModelSelectorView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        setAllCaps(false);
        refreshLabel();
        setOnClickListener(v -> {
            if (!downloading) showQualityMenu();
        });
    }

    private void refreshLabel() {
        FaceSwapper.ModelChoice model = FaceSwapper.getSelectedModel(getContext());
        IdentityStrengthSettings.Level strength = IdentityStrengthSettings.get(getContext());
        setText("Quality: " + model.displayName + " • " + strength.displayName + " identity");
    }

    private void showQualityMenu() {
        String[] items = {
            "Face swap model\n" + FaceSwapper.getSelectedModel(getContext()).displayName,
            "Identity strength\n" + IdentityStrengthSettings.get(getContext()).displayName
        };
        new AlertDialog.Builder(getContext())
            .setTitle("Quality settings")
            .setItems(items, (dialog, which) -> {
                if (which == 0) showModelSelector();
                else showIdentitySelector();
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void showModelSelector() {
        FaceSwapper.ModelChoice[] choices = FaceSwapper.ModelChoice.values();
        FaceSwapper.ModelChoice current = FaceSwapper.getSelectedModel(getContext());
        String[] labels = new String[choices.length];
        int checked = 0;
        for (int i = 0; i < choices.length; i++) {
            labels[i] = choices[i].displayName + "\n" + choices[i].description;
            if (choices[i] == current) checked = i;
        }

        final int[] selected = {checked};
        new AlertDialog.Builder(getContext())
            .setTitle("Choose face swap model")
            .setSingleChoiceItems(labels, checked, (dialog, which) -> selected[0] = which)
            .setPositiveButton("Use model", (dialog, which) -> selectAndPrepare(choices[selected[0]]))
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void showIdentitySelector() {
        IdentityStrengthSettings.Level[] levels = IdentityStrengthSettings.Level.values();
        IdentityStrengthSettings.Level current = IdentityStrengthSettings.get(getContext());
        String[] labels = new String[levels.length];
        int checked = 0;
        for (int i = 0; i < levels.length; i++) {
            labels[i] = levels[i].displayName + "\n" + levels[i].description;
            if (levels[i] == current) checked = i;
        }

        final int[] selected = {checked};
        new AlertDialog.Builder(getContext())
            .setTitle("Identity strength")
            .setSingleChoiceItems(labels, checked, (dialog, which) -> selected[0] = which)
            .setPositiveButton("Use strength", (dialog, which) -> {
                IdentityStrengthSettings.set(getContext(), levels[selected[0]]);
                refreshLabel();
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void selectAndPrepare(FaceSwapper.ModelChoice choice) {
        FaceSwapper.ModelChoice current = FaceSwapper.getSelectedModel(getContext());
        if (choice == current) {
            refreshLabel();
            return;
        }

        downloading = true;
        setEnabled(false);
        setText("Preparing " + choice.displayName + "...");

        new Thread(() -> {
            try {
                ModelDownloader downloader = new ModelDownloader(getContext());
                downloader.setCallback(new ModelDownloader.DownloadCallback() {
                    @Override
                    public void onProgress(String modelName, int progress) {
                        mainHandler.post(() -> setText("Downloading " + choice.displayName + " • " + progress + "%"));
                    }

                    @Override
                    public void onComplete(String modelName) {
                        // SimSwap has a second converter model, so completion is handled after all files are ready.
                    }

                    @Override
                    public void onError(String modelName, String error) {
                        // The thrown exception below presents the final error once.
                    }
                });

                if (choice == FaceSwapper.ModelChoice.HYPERSWAP_1C) {
                    downloader.getModelFile(ModelDownloader.HYPERSWAP_1C_MODEL);
                } else if (choice == FaceSwapper.ModelChoice.SIMSWAP_512) {
                    downloader.getModelFile(ModelDownloader.SIMSWAP_512_MODEL);
                    mainHandler.post(() -> setText("Preparing SimSwap embedding converter..."));
                    downloader.getModelFile(ModelDownloader.CROSSFACE_SIMSWAP_MODEL);
                } else {
                    downloader.getModelFile(ModelDownloader.HYPERSWAP_MODEL);
                }

                // Commit the selection only after every required file has passed integrity validation.
                FaceSwapper.setSelectedModel(getContext(), choice);
                mainHandler.post(() -> {
                    downloading = false;
                    setEnabled(true);
                    refreshLabel();
                    Toast.makeText(getContext(), choice.displayName + " is ready", Toast.LENGTH_SHORT).show();
                });
            } catch (Exception e) {
                mainHandler.post(() -> {
                    downloading = false;
                    setEnabled(true);
                    refreshLabel();
                    Toast.makeText(getContext(), "Model download failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }, "swapper-model-download").start();
    }
}
