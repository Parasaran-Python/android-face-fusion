package com.pv.androidfacefusion;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Unit tests for SavedFace data model.
 */
public class SavedFaceTest {

    @Test
    public void testSavedFaceConstructorAndGetters() {
        String id = "face-123";
        String name = "Test Person";
        String imagePath = "/data/user/0/com.pv.androidfacefusion/files/face-123.jpg";
        float[] embedding = new float[]{0.1f, 0.2f, 0.3f, -0.4f, 0.5f};
        long timestamp = 1700000000000L;

        SavedFace face = new SavedFace(id, name, imagePath, embedding, timestamp);

        assertNotNull(face);
        assertEquals(id, face.getId());
        assertEquals(name, face.getName());
        assertEquals(imagePath, face.getImagePath());
        assertArrayEquals(embedding, face.getEmbedding(), 0.0001f);
        assertEquals(timestamp, face.getTimestamp());
    }

    @Test
    public void testSetName() {
        SavedFace face = new SavedFace("id-1", "Original Name", "/path/test.jpg", new float[]{1.0f}, 1000L);
        assertEquals("Original Name", face.getName());

        face.setName("Updated Name");
        assertEquals("Updated Name", face.getName());
    }
}
