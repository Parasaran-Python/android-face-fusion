package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

import androidx.appcompat.widget.AppCompatImageView;

public class ZoomableImageView extends AppCompatImageView {

    private static final float MIN_SCALE = 1.0f;
    private static final float MAX_SCALE = 5.0f;

    private final Matrix matrix = new Matrix();
    private final float[] matrixValues = new float[9];

    private ScaleGestureDetector scaleDetector;
    private GestureDetector gestureDetector;

    private float currentScale = 1.0f;
    private boolean isReady = false;

    // For pan tracking
    private final PointF lastTouch = new PointF();
    private int activePointerId = -1;
    private boolean isPanning = false;

    public ZoomableImageView(Context context) {
        super(context);
        init(context);
    }

    public ZoomableImageView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    public ZoomableImageView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init(context);
    }

    private void init(Context context) {
        setScaleType(ScaleType.MATRIX);

        scaleDetector = new ScaleGestureDetector(context, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(ScaleGestureDetector detector) {
                float scaleFactor = detector.getScaleFactor();
                float newScale = currentScale * scaleFactor;

                if (newScale >= MIN_SCALE && newScale <= MAX_SCALE) {
                    matrix.postScale(scaleFactor, scaleFactor, detector.getFocusX(), detector.getFocusY());
                    currentScale = newScale;
                    clampTranslation();
                    setImageMatrix(matrix);
                }
                return true;
            }
        });

        gestureDetector = new GestureDetector(context, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onDoubleTap(MotionEvent e) {
                if (currentScale > 1.1f) {
                    // Reset to fit
                    resetToFit();
                } else {
                    // Zoom to 3x at tap point
                    float targetScale = 3.0f;
                    float scaleFactor = targetScale / currentScale;
                    matrix.postScale(scaleFactor, scaleFactor, e.getX(), e.getY());
                    currentScale = targetScale;
                    clampTranslation();
                    setImageMatrix(matrix);
                }
                return true;
            }
        });
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        if (w > 0 && h > 0) {
            resetToFit();
        }
    }

    @Override
    public void setImageDrawable(Drawable drawable) {
        super.setImageDrawable(drawable);
        isReady = false;
        if (drawable != null && getWidth() > 0 && getHeight() > 0) {
            resetToFit();
        }
    }

    private void resetToFit() {
        Drawable drawable = getDrawable();
        if (drawable == null) return;

        int dWidth = drawable.getIntrinsicWidth();
        int dHeight = drawable.getIntrinsicHeight();
        int vWidth = getWidth();
        int vHeight = getHeight();

        if (dWidth <= 0 || dHeight <= 0 || vWidth <= 0 || vHeight <= 0) return;

        float scale = Math.min((float) vWidth / dWidth, (float) vHeight / dHeight);
        float dx = (vWidth - dWidth * scale) / 2f;
        float dy = (vHeight - dHeight * scale) / 2f;

        matrix.reset();
        matrix.postScale(scale, scale);
        matrix.postTranslate(dx, dy);
        currentScale = 1.0f;
        isReady = true;

        setImageMatrix(matrix);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        scaleDetector.onTouchEvent(event);
        gestureDetector.onTouchEvent(event);

        if (scaleDetector.isInProgress()) {
            return true;
        }

        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                activePointerId = event.getPointerId(0);
                lastTouch.set(event.getX(), event.getY());
                isPanning = false;
                break;

            case MotionEvent.ACTION_MOVE:
                if (activePointerId != -1 && currentScale > 1.05f) {
                    int pointerIndex = event.findPointerIndex(activePointerId);
                    if (pointerIndex >= 0) {
                        float dx = event.getX(pointerIndex) - lastTouch.x;
                        float dy = event.getY(pointerIndex) - lastTouch.y;

                        if (!isPanning && (Math.abs(dx) > 10 || Math.abs(dy) > 10)) {
                            isPanning = true;
                        }

                        if (isPanning) {
                            matrix.postTranslate(dx, dy);
                            clampTranslation();
                            setImageMatrix(matrix);
                            lastTouch.set(event.getX(pointerIndex), event.getY(pointerIndex));
                        }
                    }
                }
                break;

            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                activePointerId = -1;
                isPanning = false;
                break;

            case MotionEvent.ACTION_POINTER_UP:
                int pointerIndex = event.getActionIndex();
                int pointerId = event.getPointerId(pointerIndex);
                if (pointerId == activePointerId) {
                    int newIndex = pointerIndex == 0 ? 1 : 0;
                    if (newIndex < event.getPointerCount()) {
                        activePointerId = event.getPointerId(newIndex);
                        lastTouch.set(event.getX(newIndex), event.getY(newIndex));
                    } else {
                        activePointerId = -1;
                    }
                }
                break;
        }

        return true;
    }

    private void clampTranslation() {
        Drawable drawable = getDrawable();
        if (drawable == null || !isReady) return;

        int dWidth = drawable.getIntrinsicWidth();
        int dHeight = drawable.getIntrinsicHeight();
        int vWidth = getWidth();
        int vHeight = getHeight();

        matrix.getValues(matrixValues);
        float scaleX = matrixValues[Matrix.MSCALE_X];
        float transX = matrixValues[Matrix.MTRANS_X];
        float transY = matrixValues[Matrix.MTRANS_Y];

        float scaledWidth = dWidth * scaleX;
        float scaledHeight = dHeight * scaleX;

        float dx = 0, dy = 0;

        if (scaledWidth <= vWidth) {
            dx = (vWidth - scaledWidth) / 2f - transX;
        } else {
            if (transX > 0) dx = -transX;
            else if (transX + scaledWidth < vWidth) dx = vWidth - transX - scaledWidth;
        }

        if (scaledHeight <= vHeight) {
            dy = (vHeight - scaledHeight) / 2f - transY;
        } else {
            if (transY > 0) dy = -transY;
            else if (transY + scaledHeight < vHeight) dy = vHeight - transY - scaledHeight;
        }

        if (dx != 0 || dy != 0) {
            matrix.postTranslate(dx, dy);
        }
    }
}
