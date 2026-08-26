package com.pv.androidfacefusion;

import android.content.Context;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

/**
 * Handles downloading and caching of ONNX models with resumable downloading,
 * automatic retries, HTTP redirect tracking, and mirror fallback support.
 */
public class ModelDownloader {
    private static final String TAG = "ModelDownloader";

    public static final String DET_MODEL = "det_10g.onnx";
    // Use FaceFusion's exact ArcFace recognizer and a distinct cache filename so
    // existing installs cannot silently reuse the older InsightFace download.
    public static final String REC_MODEL = "arcface_w600k_r50.onnx";
    public static final String HYPERSWAP_MODEL = "hyperswap_1b_256.onnx";
    public static final String INSWAPPER_MODEL = "inswapper_128.onnx";

    private static final List<String> DET_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/leonelhs/insightface/resolve/main/det_10g.onnx"
    );

    private static final List<String> REC_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx?download=true"
    );

    // Primary Android upgrade: FaceFusion HyperSwap 1b 256 (~403 MB).
    private static final List<String> HYPERSWAP_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx?download=true"
    );

    // Preserved as an automatic compatibility fallback.
    private static final List<String> INSWAPPER_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/leonelhs/insightface/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    );

    public interface DownloadCallback {
        void onProgress(String modelName, int progress);
        void onComplete(String modelName);
        void onError(String modelName, String error);
    }

    private final Context context;
    private DownloadCallback callback;

    public ModelDownloader(Context context) {
        this.context = context.getApplicationContext();
    }

    public void setCallback(DownloadCallback callback) {
        this.callback = callback;
    }

    public File getModelFile(String modelName) throws Exception {
        File modelFile = new File(context.getFilesDir(), modelName);

        if (modelFile.exists() && modelFile.length() > 0) {
            long minExpectedSize = getMinExpectedSize(modelName);
            if (modelFile.length() >= minExpectedSize) {
                Log.d(TAG, modelName + " already exists in cache (" + (modelFile.length() / (1024 * 1024)) + " MB)");
                return modelFile;
            }
            Log.w(TAG, modelName + " exists but incomplete (" + modelFile.length() + " bytes), resuming download...");
        }

        List<String> urls = getUrlsForModel(modelName);
        if (urls == null || urls.isEmpty()) {
            throw new Exception("Unknown model: " + modelName);
        }

        Exception lastException = null;
        for (String url : urls) {
            try {
                Log.d(TAG, "Downloading " + modelName + " from " + url);
                downloadModelWithRetry(url, modelFile, modelName);
                return modelFile;
            } catch (Exception e) {
                Log.e(TAG, "Failed downloading from " + url + ": " + e.getMessage());
                lastException = e;
            }
        }

        if (callback != null) {
            callback.onError(modelName, lastException != null ? lastException.getMessage() : "Unknown error");
        }
        throw new Exception("Failed to download " + modelName + " after trying all mirrors: "
            + (lastException != null ? lastException.getMessage() : "Unknown error"));
    }

    private long getMinExpectedSize(String modelName) {
        switch (modelName) {
            case DET_MODEL: return 10 * 1024 * 1024L;
            case REC_MODEL: return 100 * 1024 * 1024L;
            case HYPERSWAP_MODEL: return 380 * 1024 * 1024L;
            case INSWAPPER_MODEL: return 500 * 1024 * 1024L;
            default: return 1L;
        }
    }

    /**
     * Startup requires detector + FaceFusion ArcFace recognizer + HyperSwap.
     * INSwapper is downloaded only if HyperSwap initialization fails.
     */
    public boolean areAllModelsDownloaded() {
        return isModelDownloaded(DET_MODEL)
            && isModelDownloaded(REC_MODEL)
            && isModelDownloaded(HYPERSWAP_MODEL);
    }

    private boolean isModelDownloaded(String modelName) {
        File file = new File(context.getFilesDir(), modelName);
        return file.exists() && file.length() >= getMinExpectedSize(modelName);
    }

    public long getTotalModelSize() {
        long total = 0L;
        for (String modelName : new String[]{DET_MODEL, REC_MODEL, HYPERSWAP_MODEL, INSWAPPER_MODEL}) {
            File file = new File(context.getFilesDir(), modelName);
            if (file.exists()) total += file.length();
        }
        return total / (1024 * 1024);
    }

    public void clearCache() {
        for (String modelName : new String[]{DET_MODEL, REC_MODEL, HYPERSWAP_MODEL, INSWAPPER_MODEL}) {
            File file = new File(context.getFilesDir(), modelName);
            if (file.exists() && !file.delete()) {
                Log.w(TAG, "Could not delete cached model: " + modelName);
            }
        }
        // Also remove the legacy recognizer cache from older test builds.
        File legacyRecognizer = new File(context.getFilesDir(), "w600k_r50.onnx");
        if (legacyRecognizer.exists() && !legacyRecognizer.delete()) {
            Log.w(TAG, "Could not delete legacy cached recognizer: w600k_r50.onnx");
        }
    }

    private List<String> getUrlsForModel(String modelName) {
        switch (modelName) {
            case DET_MODEL: return DET_MODEL_URLS;
            case REC_MODEL: return REC_MODEL_URLS;
            case HYPERSWAP_MODEL: return HYPERSWAP_MODEL_URLS;
            case INSWAPPER_MODEL: return INSWAPPER_MODEL_URLS;
            default: return null;
        }
    }

    private void downloadModelWithRetry(String initialUrlString, File outputFile, String modelName) throws Exception {
        int maxRetries = 10;
        Exception lastException = null;

        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                downloadSingleAttempt(initialUrlString, outputFile, modelName);
                return;
            } catch (Exception e) {
                lastException = e;
                Log.w(TAG, "Download attempt " + attempt + "/" + maxRetries + " failed for " + modelName + ": " + e.getMessage());
                if (attempt < maxRetries) {
                    try {
                        Thread.sleep(2000L * attempt);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new Exception("Model download interrupted", interrupted);
                    }
                }
            }
        }

        throw new Exception("Failed after " + maxRetries + " attempts: "
            + (lastException != null ? lastException.getMessage() : "Unknown error"));
    }

    private void downloadSingleAttempt(String urlString, File outputFile, String modelName) throws Exception {
        long existingLength = outputFile.exists() ? outputFile.length() : 0L;
        HttpURLConnection connection = openConnectionWithRedirects(urlString, existingLength);
        int responseCode = connection.getResponseCode();

        if (responseCode == 416) {
            connection.disconnect();
            if (outputFile.exists()) outputFile.delete();
            existingLength = 0L;
            connection = openConnectionWithRedirects(urlString, 0L);
            responseCode = connection.getResponseCode();
        }

        boolean partial = responseCode == HttpURLConnection.HTTP_PARTIAL;
        if (responseCode != HttpURLConnection.HTTP_OK && !partial) {
            String message = connection.getResponseMessage();
            connection.disconnect();
            throw new Exception("Server returned HTTP " + responseCode + " " + message);
        }

        long totalLength = connection.getContentLengthLong();
        if (partial && totalLength > 0) totalLength += existingLength;

        try (InputStream input = new BufferedInputStream(connection.getInputStream());
             FileOutputStream output = new FileOutputStream(outputFile, partial)) {
            byte[] data = new byte[32768];
            int count;
            int lastProgress = -1;
            long downloadedBytes = partial ? existingLength : 0L;

            while ((count = input.read(data)) != -1) {
                output.write(data, 0, count);
                downloadedBytes += count;

                if (totalLength > 0 && callback != null) {
                    int progress = (int) (downloadedBytes * 100 / totalLength);
                    if (progress != lastProgress && progress % 2 == 0) {
                        callback.onProgress(modelName, progress);
                        lastProgress = progress;
                    }
                }
            }
            output.flush();

            if (outputFile.length() < getMinExpectedSize(modelName)) {
                throw new Exception("Downloaded file is smaller than expected: " + outputFile.length() + " bytes");
            }

            if (callback != null) callback.onComplete(modelName);
            Log.d(TAG, "Downloaded " + modelName + " successfully (" + downloadedBytes + " bytes)");
        } finally {
            connection.disconnect();
        }
    }

    private HttpURLConnection openConnectionWithRedirects(String urlString, long existingLength) throws Exception {
        String currentUrl = urlString;
        int redirects = 0;
        int maxRedirects = 10;

        while (redirects < maxRedirects) {
            URL url = new URL(currentUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setInstanceFollowRedirects(false);
            conn.setConnectTimeout(60000);
            conn.setReadTimeout(120000);
            conn.setRequestProperty("User-Agent", "AndroidFaceFusion/1.0");
            conn.setRequestProperty("Accept", "*/*");
            conn.setRequestProperty("Connection", "keep-alive");
            conn.setRequestProperty("Accept-Encoding", "identity");
            if (existingLength > 0) {
                conn.setRequestProperty("Range", "bytes=" + existingLength + "-");
            }

            conn.connect();
            int code = conn.getResponseCode();
            if (code == HttpURLConnection.HTTP_MOVED_PERM
                || code == HttpURLConnection.HTTP_MOVED_TEMP
                || code == HttpURLConnection.HTTP_SEE_OTHER
                || code == 307 || code == 308) {
                String location = conn.getHeaderField("Location");
                conn.disconnect();
                if (location == null || location.isEmpty()) {
                    throw new Exception("HTTP redirect with no Location header");
                }
                if (!location.startsWith("http")) {
                    URL base = new URL(currentUrl);
                    location = new URL(base, location).toExternalForm();
                }
                currentUrl = location;
                redirects++;
            } else {
                return conn;
            }
        }

        throw new Exception("Too many HTTP redirects");
    }
}
