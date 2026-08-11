package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatImageView;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Interactive ImageView that highlights detected faces and allows user to tap to select face(s).
 */
public class FaceOverlayImageView extends AppCompatImageView {

    public interface OnFaceSelectedListener {
        void onFaceSelectionChanged(Set<Integer> selectedIndices);
    }

    private List<FaceDetector.Face> faces = new ArrayList<>();
    private Set<Integer> selectedFaceIndices = new HashSet<>();
    private OnFaceSelectedListener listener;

    private Paint boxSelectedPaint;
    private Paint boxUnselectedPaint;
    private Paint badgeBgSelectedPaint;
    private Paint badgeBgUnselectedPaint;
    private Paint badgeTextPaint;

    private float density = 1.0f;

    public FaceOverlayImageView(@NonNull Context context) {
        super(context);
        init();
    }

    public FaceOverlayImageView(@NonNull Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public FaceOverlayImageView(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        density = getResources().getDisplayMetrics().density;

        // Selected face bounding box paint (bright cyan accent)
        boxSelectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxSelectedPaint.setStyle(Paint.Style.STROKE);
        boxSelectedPaint.setColor(0xFF00E5FF);
        boxSelectedPaint.setStrokeWidth(3.0f * density);

        // Unselected face bounding box paint (translucent white)
        boxUnselectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxUnselectedPaint.setStyle(Paint.Style.STROKE);
        boxUnselectedPaint.setColor(0x80FFFFFF);
        boxUnselectedPaint.setStrokeWidth(1.5f * density);

        // Selected badge background
        badgeBgSelectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        badgeBgSelectedPaint.setStyle(Paint.Style.FILL);
        badgeBgSelectedPaint.setColor(0xFF00E5FF);

        // Unselected badge background
        badgeBgUnselectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        badgeBgUnselectedPaint.setStyle(Paint.Style.FILL);
        badgeBgUnselectedPaint.setColor(0x80000000);

        // Badge text paint
        badgeTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        badgeTextPaint.setColor(0xFFFFFFFF);
        badgeTextPaint.setTextSize(12.0f * density);
        badgeTextPaint.setFakeBoldText(true);

        setClickable(true);
        setFocusable(true);
    }

    public void setFaces(List<FaceDetector.Face> detectedFaces) {
        this.faces = (detectedFaces != null) ? detectedFaces : new ArrayList<>();
        this.selectedFaceIndices.clear();
        for (int i = 0; i < this.faces.size(); i++) {
            this.selectedFaceIndices.add(i); // Default to selecting all faces
        }
        invalidate();
    }

    public void setSelectedFaceIndex(int index) {
        this.selectedFaceIndices.clear();
        if (index == -1) {
            for (int i = 0; i < faces.size(); i++) {
                this.selectedFaceIndices.add(i);
            }
        } else if (index >= 0 && index < faces.size()) {
            this.selectedFaceIndices.add(index);
        }
        invalidate();
    }

    public void toggleFaceIndex(int index) {
        if (index >= 0 && index < faces.size()) {
            if (selectedFaceIndices.contains(index)) {
                selectedFaceIndices.remove(index);
            } else {
                selectedFaceIndices.add(index);
            }
            invalidate();
        }
    }

    public void setSelectedFaceIndices(Set<Integer> indices) {
        this.selectedFaceIndices = (indices != null) ? new HashSet<>(indices) : new HashSet<>();
        invalidate();
    }

    public Set<Integer> getSelectedFaceIndices() {
        return selectedFaceIndices;
    }

    public int getSelectedFaceIndex() {
        if (selectedFaceIndices.size() == faces.size()) return -1; // All selected
        if (selectedFaceIndices.size() == 1) return selectedFaceIndices.iterator().next();
        return -1;
    }

    public List<FaceDetector.Face> getFaces() {
        return faces;
    }

    public void setOnFaceSelectedListener(OnFaceSelectedListener listener) {
        this.listener = listener;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (faces == null || faces.isEmpty() || getDrawable() == null) {
            return;
        }

        Matrix matrix = getImageMatrix();
        RectF mappedBox = new RectF();

        for (int i = 0; i < faces.size(); i++) {
            FaceDetector.Face face = faces.get(i);
            matrix.mapRect(mappedBox, face.bbox);

            boolean isSelected = selectedFaceIndices.contains(i);

            Paint boxPaint = isSelected ? boxSelectedPaint : boxUnselectedPaint;
            Paint badgeBgPaint = isSelected ? badgeBgSelectedPaint : badgeBgUnselectedPaint;

            // Draw rounded box around face
            float rx = 8.0f * density;
            float ry = 8.0f * density;
            canvas.drawRoundRect(mappedBox, rx, ry, boxPaint);

            // Draw index badge label
            String label = "Face " + (i + 1);
            float textWidth = badgeTextPaint.measureText(label);
            float badgeWidth = textWidth + 12.0f * density;
            float badgeHeight = 18.0f * density;

            float badgeLeft = mappedBox.left;
            float badgeTop = Math.max(0, mappedBox.top - badgeHeight - 2.0f * density);
            RectF badgeRect = new RectF(badgeLeft, badgeTop, badgeLeft + badgeWidth, badgeTop + badgeHeight);

            canvas.drawRoundRect(badgeRect, 4.0f * density, 4.0f * density, badgeBgPaint);

            // Set badge text color (dark gray for cyan selected badge, white for unselected)
            if (isSelected) {
                badgeTextPaint.setColor(0xFF000000);
            } else {
                badgeTextPaint.setColor(0xFFFFFFFF);
            }

            float textX = badgeLeft + 6.0f * density;
            float textY = badgeTop + badgeHeight - 4.0f * density;
            canvas.drawText(label, textX, textY, badgeTextPaint);
        }
    }

    private float downX, downY;

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                downX = event.getX();
                downY = event.getY();
                return true;

            case MotionEvent.ACTION_UP:
                float dx = Math.abs(event.getX() - downX);
                float dy = Math.abs(event.getY() - downY);
                // Touch threshold for tap vs scroll
                if (dx < 10 * density && dy < 10 * density) {
                    handleTap(event.getX(), event.getY());
                }
                return performClick();
        }
        return super.onTouchEvent(event);
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    private void handleTap(float touchX, float touchY) {
        if (faces == null || faces.isEmpty() || getDrawable() == null) {
            return;
        }

        Matrix inverse = new Matrix();
        if (getImageMatrix().invert(inverse)) {
            float[] pts = new float[]{touchX, touchY};
            inverse.mapPoints(pts);
            float bmpX = pts[0];
            float bmpY = pts[1];

            for (int i = 0; i < faces.size(); i++) {
                if (faces.get(i).bbox.contains(bmpX, bmpY)) {
                    toggleFaceIndex(i);
                    if (listener != null) {
                        listener.onFaceSelectionChanged(selectedFaceIndices);
                    }
                    return;
                }
            }
        }
    }
}
