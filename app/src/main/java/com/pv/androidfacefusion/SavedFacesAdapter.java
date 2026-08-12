package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

/**
 * Adapter for displaying saved faces in the face library recycler view.
 */
public class SavedFacesAdapter extends RecyclerView.Adapter<SavedFacesAdapter.ViewHolder> {

    public interface OnFaceActionListener {
        void onEditFaceName(SavedFace face);
        void onDeleteFace(SavedFace face);
    }

    private final Context context;
    private final List<SavedFace> faceList = new ArrayList<>();
    private final FaceLibraryManager libraryManager;
    private final OnFaceActionListener actionListener;

    public SavedFacesAdapter(Context context, FaceLibraryManager libraryManager, OnFaceActionListener actionListener) {
        this.context = context;
        this.libraryManager = libraryManager;
        this.actionListener = actionListener;
    }

    public void setSavedFaces(List<SavedFace> faces) {
        this.faceList.clear();
        if (faces != null) {
            this.faceList.addAll(faces);
        }
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_saved_face, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        SavedFace face = faceList.get(position);
        holder.nameText.setText(face.getName());

        Bitmap bitmap = libraryManager.loadFaceImage(face);
        if (bitmap != null) {
            holder.imageView.setImageBitmap(bitmap);
        } else {
            holder.imageView.setImageResource(android.R.drawable.ic_menu_camera);
        }

        holder.btnEdit.setOnClickListener(v -> {
            if (actionListener != null) {
                actionListener.onEditFaceName(face);
            }
        });

        holder.btnDelete.setOnClickListener(v -> {
            if (actionListener != null) {
                actionListener.onDeleteFace(face);
            }
        });
    }

    @Override
    public int getItemCount() {
        return faceList.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;
        TextView nameText;
        ImageButton btnEdit, btnDelete;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.savedFaceImageView);
            nameText = itemView.findViewById(R.id.savedFaceNameText);
            btnEdit = itemView.findViewById(R.id.btnEditFaceName);
            btnDelete = itemView.findViewById(R.id.btnDeleteFace);
        }
    }
}
