package com.pv.androidfacefusion;

import android.graphics.Bitmap;

import java.io.Serializable;

/**
 * Data model for a saved face in the app library.
 */
public class SavedFace implements Serializable {
    private String id;
    private String name;
    private String imagePath;
    private float[] embedding;
    private long timestamp;

    public SavedFace(String id, String name, String imagePath, float[] embedding, long timestamp) {
        this.id = id;
        this.name = name;
        this.imagePath = imagePath;
        this.embedding = embedding;
        this.timestamp = timestamp;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getImagePath() {
        return imagePath;
    }

    public float[] getEmbedding() {
        return embedding;
    }

    public long getTimestamp() {
        return timestamp;
    }
}
