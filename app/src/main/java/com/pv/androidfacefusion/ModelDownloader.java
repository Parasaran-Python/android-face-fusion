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
    
    // Model URLs - Primary and fallback mirrors
    private static final List<String> DET_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/leonelhs/insightface/resolve/main/det_10g.onnx"
    );
    
    private static final List<String> REC_MODEL_URLS = Arrays.asList(
        "https://huggingface.co/leonelhs/insightface/resolve/main/w600k_r50.onnx"
    );
    
    private static final List<String> SWAP_MODEL_URLS = Arrays.asList(
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
    
    /**
     * Get the file path for a model, downloading if necessary with resumption and retries
     */
    public File getModelFile(String modelName) throws Exception {
        File modelFile = new File(context.getFilesDir(), modelName);
        
        if (modelFile.exists() && modelFile.length() > 0) {
            long minExpectedSize = getMinExpectedSize(modelName);
            if (modelFile.length() >= minExpectedSize) {
                Log.d(TAG, modelName + " already exists in cache (" + (modelFile.length() / (1024 * 1024)) + " MB)");
                return modelFile;
            } else {
                Log.w(TAG, modelName + " exists but incomplete (" + modelFile.length() + " bytes), resuming download...");
            }
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
        
        throw new Exception("Failed to download " + modelName + " after trying all mirrors: " 
            + (lastException != null ? lastException.getMessage() : "Unknown error"));
    }

    private long getMinExpectedSize(String modelName) {
        switch (modelName) {
            case "det_10g.onnx": return 10 * 1024 * 1024L; // ~16 MB
            case "w600k_r50.onnx": return 100 * 1024 * 1024L; // ~166 MB
            case "inswapper_128.onnx": return 500 * 1024 * 1024L; // ~554 MB
            default: return 1L;
        }
    }
    
    public boolean areAllModelsDownloaded() {
        File detFile = new File(context.getFilesDir(), "det_10g.onnx");
        File recFile = new File(context.getFilesDir(), "w600k_r50.onnx");
        File swapFile = new File(context.getFilesDir(), "inswapper_128.onnx");
        
        return detFile.exists() && detFile.length() >= getMinExpectedSize("det_10g.onnx") &&
               recFile.exists() && recFile.length() >= getMinExpectedSize("w600k_r50.onnx") &&
               swapFile.exists() && swapFile.length() >= getMinExpectedSize("inswapper_128.onnx");
    }
    
    public long getTotalModelSize() {
        File detFile = new File(context.getFilesDir(), "det_10g.onnx");
        File recFile = new File(context.getFilesDir(), "w600k_r50.onnx");
        File swapFile = new File(context.getFilesDir(), "inswapper_128.onnx");
        
        long total = 0;
        if (detFile.exists()) total += detFile.length();
        if (recFile.exists()) total += recFile.length();
        if (swapFile.exists()) total += swapFile.length();
        
        return total / (1024 * 1024);
    }
    
    public void clearCache() {
        File detFile = new File(context.getFilesDir(), "det_10g.onnx");
        File recFile = new File(context.getFilesDir(), "w600k_r50.onnx");
        File swapFile = new File(context.getFilesDir(), "inswapper_128.onnx");
        
        if (detFile.exists()) detFile.delete();
        if (recFile.exists()) recFile.delete();
        if (swapFile.exists()) swapFile.delete();
    }
    
    private List<String> getUrlsForModel(String modelName) {
        switch (modelName) {
            case "det_10g.onnx": return DET_MODEL_URLS;
            case "w600k_r50.onnx": return REC_MODEL_URLS;
            case "inswapper_128.onnx": return SWAP_MODEL_URLS;
            default: return null;
        }
    }
    
    private void downloadModelWithRetry(String initialUrlString, File outputFile, String modelName) throws Exception {
        int maxRetries = 10;
        Exception lastException = null;
        
        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                downloadSingleAttempt(initialUrlString, outputFile, modelName);
                return; // Download succeeded
            } catch (Exception e) {
                lastException = e;
                Log.w(TAG, "Download attempt " + attempt + "/" + maxRetries + " failed for " + modelName + ": " + e.getMessage());
                
                if (attempt < maxRetries) {
                    try {
                        Thread.sleep(2000L * attempt); // Exponential backoff
                    } catch (InterruptedException ignored) {}
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
        
        if (responseCode == 416) { // Requested Range Not Satisfiable - invalid range or completed file
            connection.disconnect();
            if (outputFile.exists()) outputFile.delete();
            existingLength = 0L;
            connection = openConnectionWithRedirects(urlString, 0L);
            responseCode = connection.getResponseCode();
        }

        boolean isPartialContent = (responseCode == HttpURLConnection.HTTP_PARTIAL);
        if (responseCode != HttpURLConnection.HTTP_OK && !isPartialContent) {
            connection.disconnect();
            throw new Exception("Server returned HTTP " + responseCode + " " + connection.getResponseMessage());
        }
        
        long totalLength = connection.getContentLengthLong();
        if (isPartialContent) {
            totalLength += existingLength;
        }
        
        InputStream input = null;
        FileOutputStream output = null;
        
        try {
            input = new BufferedInputStream(connection.getInputStream());
            output = new FileOutputStream(outputFile, isPartialContent);
            
            byte[] data = new byte[32768]; // 32KB buffer
            int count;
            int lastProgress = -1;
            long downloadedBytes = isPartialContent ? existingLength : 0L;
            
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
            
            if (callback != null) {
                callback.onComplete(modelName);
            }
            
            Log.d(TAG, "Downloaded " + modelName + " successfully (" + downloadedBytes + " bytes)");
        } finally {
            if (output != null) {
                try { output.close(); } catch (Exception ignored) {}
            }
            if (input != null) {
                try { input.close(); } catch (Exception ignored) {}
            }
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
            conn.setInstanceFollowRedirects(false); // Manually handle redirects
            conn.setConnectTimeout(60000);
            conn.setReadTimeout(120000);
            conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
            conn.setRequestProperty("Accept", "*/*");
            conn.setRequestProperty("Connection", "keep-alive");
            conn.setRequestProperty("Accept-Encoding", "identity");
            
            if (existingLength > 0) {
                conn.setRequestProperty("Range", "bytes=" + existingLength + "-");
            }
            
            conn.connect();
            int code = conn.getResponseCode();
            
            if (code == HttpURLConnection.HTTP_MOVED_PERM ||
                code == HttpURLConnection.HTTP_MOVED_TEMP ||
                code == HttpURLConnection.HTTP_SEE_OTHER ||
                code == 307 || code == 308) {
                
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
