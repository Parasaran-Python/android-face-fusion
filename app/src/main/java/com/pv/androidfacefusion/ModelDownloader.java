package com.pv.androidfacefusion;

import android.content.Context;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Handles downloading and caching of ONNX models
 */
public class ModelDownloader {
    private static final String TAG = "ModelDownloader";
    
    // Model URLs - From leonelhs/insightface repository on HuggingFace
    private static final String DET_MODEL_URL = 
        "https://huggingface.co/leonelhs/insightface/resolve/main/det_10g.onnx";
    
    private static final String REC_MODEL_URL = 
        "https://huggingface.co/leonelhs/insightface/resolve/main/w600k_r50.onnx";
    
    private static final String SWAP_MODEL_URL = 
        "https://huggingface.co/leonelhs/insightface/resolve/main/inswapper_128.onnx";
    
    public interface DownloadCallback {
        void onProgress(String modelName, int progress);
        void onComplete(String modelName);
        void onError(String modelName, String error);
    }
    
    private Context context;
    private DownloadCallback callback;
    
    public ModelDownloader(Context context) {
        this.context = context;
    }
    
    public void setCallback(DownloadCallback callback) {
        this.callback = callback;
    }
    
    /**
     * Get the file path for a model, downloading if necessary
     */
    public File getModelFile(String modelName) throws Exception {
        File modelFile = new File(context.getFilesDir(), modelName);
        
        if (modelFile.exists()) {
            Log.d(TAG, modelName + " already exists in cache");
            return modelFile;
        }
        
        // Model doesn't exist, download it
        String url = getUrlForModel(modelName);
        if (url == null) {
            throw new Exception("Unknown model: " + modelName);
        }
        
        Log.d(TAG, "Downloading " + modelName + " from " + url);
        downloadModel(url, modelFile, modelName);
        
        return modelFile;
    }
    
    /**
     * Check if all models are downloaded
     */
    public boolean areAllModelsDownloaded() {
        File detFile = new File(context.getFilesDir(), "det_10g.onnx");
        File recFile = new File(context.getFilesDir(), "w600k_r50.onnx");
        File swapFile = new File(context.getFilesDir(), "inswapper_128.onnx");
        
        return detFile.exists() && recFile.exists() && swapFile.exists();
    }
    
    /**
     * Get total size of downloaded models in MB
     */
    public long getTotalModelSize() {
        File detFile = new File(context.getFilesDir(), "det_10g.onnx");
        File recFile = new File(context.getFilesDir(), "w600k_r50.onnx");
        File swapFile = new File(context.getFilesDir(), "inswapper_128.onnx");
        
        long total = 0;
        if (detFile.exists()) total += detFile.length();
        if (recFile.exists()) total += recFile.length();
        if (swapFile.exists()) total += swapFile.length();
        
        return total / (1024 * 1024); // Convert to MB
    }
    
    /**
     * Delete all cached models (to free space or force re-download)
     */
    public void clearCache() {
        File detFile = new File(context.getFilesDir(), "det_10g.onnx");
        File recFile = new File(context.getFilesDir(), "w600k_r50.onnx");
        File swapFile = new File(context.getFilesDir(), "inswapper_128.onnx");
        
        if (detFile.exists()) detFile.delete();
        if (recFile.exists()) recFile.delete();
        if (swapFile.exists()) swapFile.delete();
    }
    
    private String getUrlForModel(String modelName) {
        switch (modelName) {
            case "det_10g.onnx":
                return DET_MODEL_URL;
            case "w600k_r50.onnx":
                return REC_MODEL_URL;
            case "inswapper_128.onnx":
                return SWAP_MODEL_URL;
            default:
                return null;
        }
    }
    
    private void downloadModel(String urlString, File outputFile, String modelName) throws Exception {
        HttpURLConnection connection = null;
        BufferedInputStream input = null;
        FileOutputStream output = null;
        
        try {
            URL url = new URL(urlString);
            connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(30000);
            connection.setReadTimeout(30000);
            connection.connect();
            
            if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
                throw new Exception("Server returned HTTP " + connection.getResponseCode() 
                    + " " + connection.getResponseMessage());
            }
            
            int fileLength = connection.getContentLength();
            input = new BufferedInputStream(connection.getInputStream());
            output = new FileOutputStream(outputFile);
            
            byte[] data = new byte[8192];
            long total = 0;
            int count;
            int lastProgress = 0;
            
            while ((count = input.read(data)) != -1) {
                total += count;
                output.write(data, 0, count);
                
                if (fileLength > 0 && callback != null) {
                    int progress = (int) (total * 100 / fileLength);
                    if (progress != lastProgress && progress % 5 == 0) {
                        callback.onProgress(modelName, progress);
                        lastProgress = progress;
                    }
                }
            }
            
            output.flush();
            
            if (callback != null) {
                callback.onComplete(modelName);
            }
            
            Log.d(TAG, "Downloaded " + modelName + " successfully");
            
        } catch (Exception e) {
            // Clean up partial download
            if (outputFile.exists()) {
                outputFile.delete();
            }
            
            if (callback != null) {
                callback.onError(modelName, e.getMessage());
            }
            
            throw new Exception("Failed to download " + modelName + ": " + e.getMessage());
            
        } finally {
            if (output != null) {
                try {
                    output.close();
                } catch (Exception e) {
                    Log.e(TAG, "Error closing output stream", e);
                }
            }
            if (input != null) {
                try {
                    input.close();
                } catch (Exception e) {
                    Log.e(TAG, "Error closing input stream", e);
                }
            }
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
}
