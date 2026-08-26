package com.pv.androidfacefusion;

import android.content.Context;
import android.content.SharedPreferences;

/** Persistent identity-vs-target-preservation setting shared by every swapper model. */
public final class IdentityStrengthSettings {
    private static final String PREFS = "face_fusion_quality";
    private static final String KEY = "identity_strength";

    public enum Level {
        NATURAL("natural", "Natural", "Softer identity with more target lighting and texture", 1.00f, 1.00f),
        STRONG("strong", "Strong", "Recommended • clearer source identity with natural expression", 1.38f, 0.55f),
        MAX("max", "Max", "Maximum source identity with minimal target appearance transfer", 1.75f, 0.20f);

        public final String id;
        public final String displayName;
        public final String description;
        final float maskExponent;
        final float targetReferenceWeight;

        Level(String id, String displayName, String description,
              float maskExponent, float targetReferenceWeight) {
            this.id = id;
            this.displayName = displayName;
            this.description = description;
            this.maskExponent = maskExponent;
            this.targetReferenceWeight = targetReferenceWeight;
        }

        static Level fromId(String id) {
            for (Level level : values()) {
                if (level.id.equals(id)) return level;
            }
            return STRONG;
        }
    }

    private IdentityStrengthSettings() {}

    public static Level get(Context context) {
        SharedPreferences prefs = context.getApplicationContext()
            .getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        return Level.fromId(prefs.getString(KEY, Level.STRONG.id));
    }

    public static void set(Context context, Level level) {
        context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY, level.id).apply();
    }
}
