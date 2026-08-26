package com.pv.androidfacefusion;

import android.content.Context;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;

/** Downloads, validates and caches the ONNX models used by the app. */
public class ModelDownloader {
    private static final String TAG = "ModelDownloader";

    public static final String DET_MODEL = "det_10g.onnx";
    public static final String REC_MODEL = "arcface_w600k_r50.onnx";
    public static final String HYPERSWAP_MODEL = "hyperswap_1a_256.onnx";
    public static final String LANDMARKER_MODEL = "2dfan4.onnx";
    public static final String PARSER_MODEL = "bisenet_resnet_18.onnx";
    public static final String INSWAPPER_MODEL = "inswapper_128.onnx";

    private static final String REC_MODEL_SHA256 =
        "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1";
    private static final String HYPERSWAP_MODEL_SHA256 =
        "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add";
    private static final String LANDMARKER_MODEL_SHA256 =
        "678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285";
    private static final String PARSER_MODEL_SHA256 =
        "2218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9";

    private static final List<String> DET_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/leonelhs/insightface/resolve/main/det_10g.onnx"
    );
    private static final List<String> REC_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx?download=true"
    );
    private static final List<String> HYPERSWAP_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx?download=true"
    );
    private static final List<String> LANDMARKER_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.onnx?download=true"
    );
    private static final List<String> PARSER_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/facefusion/models-3.1.0/resolve/main/bisenet_resnet_18.onnx?download=true"
    );
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

        if (modelFile.exists() && modelFile.length() >= getMinExpectedSize(modelName)) {
            if (validateKnownHash(modelName, modelFile)) {
                Log.i(TAG, modelName + " cache validated (" + modelFile.length() + " bytes)");
                return modelFile;
            }
            Log.w(TAG, modelName + " failed integrity validation; deleting cached copy");
            if (!modelFile.delete()) {
                throw new Exception("Invalid cached model could not be deleted: " + modelName);
            }
        }

        List<String> urls = getUrlsForModel(modelName);
        if (urls == null || urls.isEmpty()) throw new Exception("Unknown model: " + modelName);

        Exception lastException = null;
        for (String url : urls) {
            try {
                downloadModelWithRetry(url, modelFile, modelName);
                if (!validateKnownHash(modelName, modelFile)) {
                    if (modelFile.exists()) modelFile.delete();
                    throw new Exception("Downloaded model failed SHA-256 validation: " + modelName);
                }
                return modelFile;
            } catch (Exception e) {
                Log.e(TAG, "Failed downloading " + modelName + " from " + url, e);
                lastException = e;
            }
        }

        if (callback != null) callback.onError(modelName,
            lastException != null ? lastException.getMessage() : "Unknown error");
        throw new Exception("Failed to download " + modelName + ": "
            + (lastException != null ? lastException.getMessage() : "Unknown error"));
    }

    private boolean validateKnownHash(String modelName, File file) throws Exception {
        String expected = getExpectedSha256(modelName);
        if (expected == null) return file.length() >= getMinExpectedSize(modelName);
        String actual = sha256(file);
        boolean valid = expected.equalsIgnoreCase(actual);
        if (!valid) Log.e(TAG, modelName + " SHA-256 mismatch. Expected=" + expected + " actual=" + actual);
        return valid;
    }

    private String getExpectedSha256(String modelName) {
        switch (modelName) {
            case REC_MODEL: return REC_MODEL_SHA256;
            case HYPERSWAP_MODEL: return HYPERSWAP_MODEL_SHA256;
            case LANDMARKER_MODEL: return LANDMARKER_MODEL_SHA256;
            case PARSER_MODEL: return PARSER_MODEL_SHA256;
            default: return null;
        }
    }

    private String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) digest.update(buffer, 0, read);
        }
        byte[] bytes = digest.digest();
        StringBuilder hex = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) hex.append(String.format("%02x", b & 0xff));
        return hex.toString();
    }

    private long getMinExpectedSize(String modelName) {
        switch (modelName) {
            case DET_MODEL: return 10 * 1024 * 1024L;
            case REC_MODEL: return 160 * 1024 * 1024L;
            case HYPERSWAP_MODEL: return 380 * 1024 * 1024L;
            case LANDMARKER_MODEL: return 90 * 1024 * 1024L;
            case PARSER_MODEL: return 50 * 1024 * 1024L;
            case INSWAPPER_MODEL: return 500 * 1024 * 1024L;
            default: return 1L;
        }
    }

    /** Base models required before the UI becomes usable. Quality models load lazily on first swap. */
    public boolean areAllModelsDownloaded() {
        try {
            return isModelDownloaded(DET_MODEL)
                && isModelDownloaded(REC_MODEL)
                && isModelDownloaded(HYPERSWAP_MODEL);
        } catch (Exception e) {
            Log.w(TAG, "Model integrity check failed", e);
            return false;
        }
    }

    private boolean isModelDownloaded(String modelName) throws Exception {
        File file = new File(context.getFilesDir(), modelName);
        return file.exists()
            && file.length() >= getMinExpectedSize(modelName)
            && validateKnownHash(modelName, file);
    }

    public long getTotalModelSize() {
        long total = 0L;
        for (String modelName : new String[]{DET_MODEL, REC_MODEL, HYPERSWAP_MODEL,
            LANDMARKER_MODEL, PARSER_MODEL, INSWAPPER_MODEL}) {
            File file = new File(context.getFilesDir(), modelName);
            if (file.exists()) total += file.length();
        }
        return total / (1024 * 1024);
    }

    public void clearCache() {
        for (String modelName : new String[]{DET_MODEL, REC_MODEL, HYPERSWAP_MODEL,
            LANDMARKER_MODEL, PARSER_MODEL, INSWAPPER_MODEL,
            "w600k_r50.onnx", "hyperswap_1b_256.onnx"}) {
            File file = new File(context.getFilesDir(), modelName);
            if (file.exists() && !file.delete()) Log.w(TAG, "Could not delete cached model: " + modelName);
        }
    }

    private List<String> getUrlsForModel(String modelName) {
        switch (modelName) {
            case DET_MODEL: return DET_MODEL_URLS;
            case REC_MODEL: return REC_MODEL_URLS;
            case HYPERSWAP_MODEL: return HYPERSWAP_MODEL_URLS;
            case LANDMARKER_MODEL: return LANDMARKER_MODEL_URLS;
            case PARSER_MODEL: return PARSER_MODEL_URLS;
            case INSWAPPER_MODEL: return INSWAPPER_MODEL_URLS;
            default: return null;
        }
    }

    private void downloadModelWithRetry(String initialUrl, File outputFile, String modelName) throws Exception {
        Exception lastException = null;
        for (int attempt = 1; attempt <= 5; attempt++) {
            try {
                downloadSingleAttempt(initialUrl, outputFile, modelName);
                return;
            } catch (Exception e) {
                lastException = e;
                Log.w(TAG, "Download attempt " + attempt + "/5 failed for " + modelName + ": " + e.getMessage());
                if (attempt < 5) Thread.sleep(1500L * attempt);
            }
        }
        throw new Exception("Failed after 5 attempts: "
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
        } finally {
            connection.disconnect();
        }
    }

    private HttpURLConnection openConnectionWithRedirects(String urlString, long existingLength) throws Exception {
        String currentUrl = urlString;
        for (int redirects = 0; redirects < 10; redirects++) {
            URL url = new URL(currentUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setInstanceFollowRedirects(false);
            conn.setConnectTimeout(60000);
            conn.setReadTimeout(120000);
            conn.setRequestProperty("User-Agent", "AndroidFaceFusion/1.0");
            conn.setRequestProperty("Accept", "*/*");
            conn.setRequestProperty("Connection", "keep-alive");
            conn.setRequestProperty("Accept-Encoding", "identity");
            if (existingLength > 0) conn.setRequestProperty("Range", "bytes=" + existingLength + "-");

            conn.connect();
            int code = conn.getResponseCode();
            if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
                String location = conn.getHeaderField("Location");
                conn.disconnect();
                if (location == null || location.isEmpty()) throw new Exception("Redirect without Location header");
                if (!location.startsWith("http")) location = new URL(new URL(currentUrl), location).toExternalForm();
                currentUrl = location;
            } else {
                return conn;
            }
        }
        throw new Exception("Too many HTTP redirects");
    }
}
