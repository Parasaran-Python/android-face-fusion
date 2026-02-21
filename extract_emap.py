#!/usr/bin/env python3
"""
Extract EMAP transformation matrix from inswapper_128.onnx model
This matrix is critical for proper face swapping and must be extracted from the model.

Python code reference (inswapper.py line 18):
    self.emap = numpy_helper.to_array(graph.initializer[-1])
"""

import onnx
from onnx import numpy_helper
import numpy as np
import struct
import os

def extract_emap(model_path, output_path):
    """
    Extract EMAP from ONNX model and save as binary file for Android app
    """
    print(f"Loading ONNX model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph
    
    print(f"Model has {len(graph.initializer)} initializers")
    
    # Get the last initializer (EMAP)
    emap_tensor = graph.initializer[-1]
    print(f"Last initializer name: {emap_tensor.name}")
    print(f"Last initializer dims: {emap_tensor.dims}")
    
    # Convert to numpy array
    emap = numpy_helper.to_array(emap_tensor)
    print(f"EMAP shape: {emap.shape}")
    print(f"EMAP dtype: {emap.dtype}")
    print(f"EMAP min: {emap.min()}, max: {emap.max()}, mean: {emap.mean()}")
    
    # Verify it's the right shape (should be 512x512)
    if emap.shape != (512, 512):
        print(f"WARNING: Expected shape (512, 512) but got {emap.shape}")
        print("This might not be the correct EMAP!")
        return False
    
    # Save as binary file (float32, row-major order)
    print(f"Saving EMAP to: {output_path}")
    with open(output_path, 'wb') as f:
        # Write shape info first (2 ints: rows, cols)
        f.write(struct.pack('ii', emap.shape[0], emap.shape[1]))
        # Write the matrix data as float32
        emap.astype(np.float32).tofile(f)
    
    file_size = os.path.getsize(output_path)
    expected_size = 8 + 512 * 512 * 4  # 8 bytes header + 512*512*4 bytes data
    print(f"Output file size: {file_size} bytes (expected: {expected_size} bytes)")
    
    if file_size == expected_size:
        print("✅ EMAP extracted successfully!")
        return True
    else:
        print("❌ File size mismatch!")
        return False

def verify_emap(emap_path):
    """
    Verify the extracted EMAP file
    """
    print(f"\nVerifying EMAP file: {emap_path}")
    
    with open(emap_path, 'rb') as f:
        # Read shape
        rows, cols = struct.unpack('ii', f.read(8))
        print(f"EMAP shape from file: ({rows}, {cols})")
        
        # Read matrix data
        emap = np.fromfile(f, dtype=np.float32).reshape((rows, cols))
        print(f"EMAP min: {emap.min()}, max: {emap.max()}, mean: {emap.mean()}")
        print(f"First 5x5 corner:\n{emap[:5, :5]}")
        
        # Check if it's NOT an identity matrix
        identity = np.eye(rows, cols, dtype=np.float32)
        is_identity = np.allclose(emap, identity)
        print(f"Is identity matrix: {is_identity}")
        
        if is_identity:
            print("⚠️  WARNING: EMAP appears to be an identity matrix!")
            print("This should NOT happen for inswapper_128.onnx")
        else:
            print("✅ EMAP is a proper learned transformation matrix")
        
        return not is_identity

def download_model(url, output_path):
    """
    Download the model from HuggingFace if not found locally
    """
    print(f"Downloading model from HuggingFace...")
    print(f"URL: {url}")
    print("This will take a while (model is ~555 MB)...")
    
    try:
        import urllib.request
        
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 100 / totalsize
                s = f"\r{percent:5.1f}% {readsofar // (1024*1024):d} MB / {totalsize // (1024*1024):d} MB"
                print(s, end='')
            else:
                print(f"\rDownloaded {readsofar // (1024*1024):d} MB", end='')
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print()  # New line after progress
        print(f"✅ Model downloaded successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == '__main__':
    # Try to find the model file
    model_paths = [
        'inswapper_128.onnx',
        '../inswapper_128.onnx',
        'models/inswapper_128.onnx',
        'D:/Code/ONNX_learn/FaceFusion/inswapper_128.onnx',
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Found model at: {path}")
            break
    
    if model_path is None:
        print("Model not found locally. Downloading from HuggingFace...")
        model_url = "https://huggingface.co/leonelhs/insightface/resolve/main/inswapper_128.onnx"
        model_path = "inswapper_128.onnx"
        
        if not download_model(model_url, model_path):
            print("\nERROR: Could not download model.")
            print("\nManual download options:")
            print("1. Download from: https://huggingface.co/leonelhs/insightface/resolve/main/inswapper_128.onnx")
            print("2. Or pull from Android device: adb pull /data/data/com.pv.androidfacefusion/files/inswapper_128.onnx")
            exit(1)
    
    output_path = 'emap.bin'
    
    # Extract EMAP
    success = extract_emap(model_path, output_path)
    
    if success:
        # Verify the extraction
        verify_emap(output_path)
        
        print(f"\n{'='*60}")
        print("NEXT STEPS:")
        print("1. Copy emap.bin to Android app: app/src/main/assets/")
        print("2. Update FaceSwapper.java to load emap.bin instead of using identity matrix")
        print(f"{'='*60}")
