package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Manages persistent storage of saved faces in app data directory.
 */
public class FaceLibraryManager {
    private static final String TAG = "FaceLibraryManager";
    private static final String FOLDER_NAME = "saved_faces";
    private static final String METADATA_FILE = "saved_faces.json";

    private final Context context;
    private final File libraryDir;
    private final File metadataFile;

    public FaceLibraryManager(Context context) {
        this.context = context.getApplicationContext();
        this.libraryDir = new File(this.context.getFilesDir(), FOLDER_NAME);
        if (!libraryDir.exists()) {
            libraryDir.mkdirs();
        }
        this.metadataFile = new File(libraryDir, METADATA_FILE);
    }

    public synchronized List<SavedFace> getSavedFaces() {
        List<SavedFace> faces = new ArrayList<>();
        if (!metadataFile.exists()) {
            return faces;
        }

        try (FileInputStream fis = new FileInputStream(metadataFile);
             InputStreamReader reader = new InputStreamReader(fis, StandardCharsets.UTF_8)) {
            
            StringBuilder sb = new StringBuilder();
            char[] buffer = new char[1024];
            int read;
            while ((read = reader.read(buffer)) != -1) {
                sb.append(buffer, 0, read);
            }

            JSONArray jsonArray = new JSONArray(sb.toString());
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject obj = jsonArray.getJSONObject(i);
                String id = obj.getString("id");
                String name = obj.getString("name");
                String imagePath = obj.getString("imagePath");
                long timestamp = obj.optLong("timestamp", System.currentTimeMillis());

                JSONArray embArray = obj.getJSONArray("embedding");
                float[] embedding = new float[embArray.length()];
                for (int j = 0; j < embArray.length(); j++) {
                    embedding[j] = (float) embArray.getDouble(j);
                }

                faces.add(new SavedFace(id, name, imagePath, embedding, timestamp));
            }
        } catch (Exception e) {
            Log.e(TAG, "Error reading saved faces metadata", e);
        }

        return faces;
    }

    public synchronized SavedFace saveFace(String name, Bitmap faceCrop, float[] embedding) {
        String id = UUID.randomUUID().toString();
        String imageName = "face_" + id + ".jpg";
        File imageFile = new File(libraryDir, imageName);

        try (FileOutputStream out = new FileOutputStream(imageFile)) {
            faceCrop.compress(Bitmap.CompressFormat.JPEG, 90, out);
        } catch (Exception e) {
            Log.e(TAG, "Error saving face crop bitmap", e);
            return null;
        }

        SavedFace savedFace = new SavedFace(
            id,
            name,
            imageFile.getAbsolutePath(),
            embedding,
            System.currentTimeMillis()
        );

        List<SavedFace> currentList = getSavedFaces();
        currentList.add(savedFace);
        persistMetadata(currentList);

        return savedFace;
    }

    public synchronized boolean updateFaceName(String id, String newName) {
        List<SavedFace> currentList = getSavedFaces();
        boolean found = false;
        for (SavedFace face : currentList) {
            if (face.getId().equals(id)) {
                face.setName(newName);
                found = true;
                break;
            }
        }
        if (found) {
            persistMetadata(currentList);
        }
        return found;
    }

    public synchronized boolean deleteFace(String id) {
        List<SavedFace> currentList = getSavedFaces();
        SavedFace target = null;
        for (SavedFace face : currentList) {
            if (face.getId().equals(id)) {
                target = face;
                break;
            }
        }

        if (target != null) {
            currentList.remove(target);
            File imgFile = new File(target.getImagePath());
            if (imgFile.exists()) {
                imgFile.delete();
            }
            persistMetadata(currentList);
            return true;
        }
        return false;
    }

    public Bitmap loadFaceImage(SavedFace face) {
        if (face == null || face.getImagePath() == null) return null;
        File imgFile = new File(face.getImagePath());
        if (imgFile.exists()) {
            return BitmapFactory.decodeFile(imgFile.getAbsolutePath());
        }
        return null;
    }

    private void persistMetadata(List<SavedFace> faces) {
        try {
            JSONArray jsonArray = new JSONArray();
            for (SavedFace face : faces) {
                JSONObject obj = new JSONObject();
                obj.put("id", face.getId());
                obj.put("name", face.getName());
                obj.put("imagePath", face.getImagePath());
                obj.put("timestamp", face.getTimestamp());

                JSONArray embArray = new JSONArray();
                for (float val : face.getEmbedding()) {
                    embArray.put((double) val);
                }
                obj.put("embedding", embArray);

                jsonArray.put(obj);
            }

            try (FileOutputStream fos = new FileOutputStream(metadataFile);
                 OutputStreamWriter writer = new OutputStreamWriter(fos, StandardCharsets.UTF_8)) {
                writer.write(jsonArray.toString(2));
            }
        } catch (Exception e) {
            Log.e(TAG, "Error persisting metadata", e);
        }
    }
}
