package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adapter for mapping each detected target face to a saved library face.
 */
public class TargetFaceMappingAdapter extends RecyclerView.Adapter<TargetFaceMappingAdapter.ViewHolder> {

    private final Context context;
    private Bitmap targetBitmap;
    private final List<FaceDetector.Face> targetFaces = new ArrayList<>();
    private final List<SavedFace> savedFaces = new ArrayList<>();
    private final Map<Integer, SavedFace> selectedMapping = new HashMap<>();
    private final Map<Integer, Bitmap> cropCache = new HashMap<>();

    public TargetFaceMappingAdapter(Context context) {
        this.context = context;
    }

    public void setData(Bitmap targetBitmap, List<FaceDetector.Face> faces, List<SavedFace> savedFaces) {
        this.targetBitmap = targetBitmap;
        this.targetFaces.clear();
        if (faces != null) {
            this.targetFaces.addAll(faces);
        }
        this.savedFaces.clear();
        if (savedFaces != null) {
            this.savedFaces.addAll(savedFaces);
        }
        this.selectedMapping.clear();
        clearCropCache();
        notifyDataSetChanged();
    }

    public void updateSavedFaces(List<SavedFace> newSavedFaces) {
        this.savedFaces.clear();
        if (newSavedFaces != null) {
            this.savedFaces.addAll(newSavedFaces);
        }
        notifyDataSetChanged();
    }

    private void clearCropCache() {
        for (Bitmap bmp : cropCache.values()) {
            if (bmp != null && !bmp.isRecycled()) {
                bmp.recycle();
            }
        }
        cropCache.clear();
    }

    public Map<Integer, SavedFace> getSelectedMapping() {
        return selectedMapping;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_target_face_mapping, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        FaceDetector.Face face = targetFaces.get(position);
        holder.label.setText("Target Face " + (position + 1));

        // Use cached crop or compute new crop thumbnail
        Bitmap crop = cropCache.get(position);
        if (crop == null && targetBitmap != null && face.bbox != null) {
            crop = cropFaceThumbnail(targetBitmap, face.bbox);
            if (crop != null) {
                cropCache.put(position, crop);
            }
        }

        if (crop != null) {
            holder.cropImageView.setImageBitmap(crop);
        } else {
            holder.cropImageView.setImageResource(android.R.drawable.ic_menu_camera);
        }

        // Detach old listener to prevent unwanted triggers during binding
        holder.spinner.setOnItemSelectedListener(null);

        // Prepare spinner dropdown options
        List<String> options = new ArrayList<>();
        options.add("Keep Original (No Swap)");
        for (SavedFace sf : savedFaces) {
            options.add("👤 " + sf.getName());
        }

        ArrayAdapter<String> spinnerAdapter = new ArrayAdapter<>(
            context,
            android.R.layout.simple_spinner_item,
            options
        );
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        holder.spinner.setAdapter(spinnerAdapter);

        // Pre-select if previously selected
        SavedFace currentlySelected = selectedMapping.get(position);
        int defaultSelection = 0;
        if (currentlySelected != null) {
            for (int i = 0; i < savedFaces.size(); i++) {
                if (savedFaces.get(i).getId().equals(currentlySelected.getId())) {
                    defaultSelection = i + 1;
                    break;
                }
            }
        }
        holder.spinner.setSelection(defaultSelection);

        int pos = position;
        holder.spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int selectedPos, long id) {
                if (selectedPos == 0) {
                    selectedMapping.remove(pos);
                } else if (selectedPos - 1 < savedFaces.size()) {
                    selectedMapping.put(pos, savedFaces.get(selectedPos - 1));
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                selectedMapping.remove(pos);
            }
        });
    }

    @Override
    public int getItemCount() {
        return targetFaces.size();
    }

    private Bitmap cropFaceThumbnail(Bitmap src, RectF bbox) {
        try {
            int left = Math.max(0, (int) bbox.left);
            int top = Math.max(0, (int) bbox.top);
            int right = Math.min(src.getWidth(), (int) bbox.right);
            int bottom = Math.min(src.getHeight(), (int) bbox.bottom);

            int width = right - left;
            int height = bottom - top;

            if (width > 0 && height > 0) {
                return Bitmap.createBitmap(src, left, top, width, height);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView cropImageView;
        TextView label;
        Spinner spinner;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            cropImageView = itemView.findViewById(R.id.targetFaceCropImageView);
            label = itemView.findViewById(R.id.targetFaceLabel);
            spinner = itemView.findViewById(R.id.savedFaceSpinner);
        }
    }
}
