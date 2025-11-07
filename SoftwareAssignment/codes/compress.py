import numpy as np
from PIL import Image
import ctypes
import os
import matplotlib.pyplot as plt

# ===== Load C Shared Library =====
LIB_NAME = "./svd_lib.so"   # or "tsvd.dll" on Windows
lib = ctypes.CDLL(LIB_NAME)

lib.truncated_svd.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # Ak_out
    ctypes.POINTER(ctypes.c_double)   # sigma_out
]
lib.truncated_svd.restype = None


def call_truncated_svd(A, k):
    """Call the C SVD function"""
    m, n = A.shape
    A_flat = np.ascontiguousarray(A.astype(np.float64)).ravel()
    Ak_out = np.zeros(m * n, dtype=np.float64)
    sigma_out = np.zeros(k, dtype=np.float64)

    A_ptr = A_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    Ak_ptr = Ak_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sigma_ptr = sigma_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.truncated_svd(A_ptr, m, n, k, Ak_ptr, sigma_ptr)
    Ak = Ak_out.reshape((m, n))
    return Ak, sigma_out


def frobenius_norm(A, B):
    """Compute Frobenius norm ||A - B||_F"""
    return np.linalg.norm(A - B, 'fro')


def save_same_format(matrix, input_path, k):
    """Save reconstructed image with same extension and format as input"""
    # Extract base name and extension (e.g. "photo", ".jpg")
    base, ext = os.path.splitext(input_path)

    # Determine image format for saving (PIL uses uppercase like "JPEG", "PNG")
    format_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".tiff": "TIFF"
    }
    img_format = format_map.get(ext.lower(), "PNG")  # fallback to PNG

    # Build output path
    output_path = f"{base}_recon_k{k}{ext}"

    # Clip and convert back to uint8 image
    img_arr = np.clip(matrix, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    img.save(output_path, format=img_format)
    print(f"✅ Saved reconstructed image: {output_path} (format: {img_format})")
    return output_path


def main():
    # === Input image (any format: jpg, png, bmp...) ===
    input_path = "globe.jpg"  # Change filename here
    img = Image.open(input_path).convert("L")  # Grayscale
    A = np.array(img, dtype=np.float64)

    ks = [5, 20, 50, 100]
    results = []

    for k in ks:
        if k > min(A.shape):
            print(f"Skipping k={k} (too large)")
            continue

        print(f"\n🔹 Computing SVD with k = {k}")
        Ak, sigma = call_truncated_svd(A, k)
        err = frobenius_norm(A, Ak)
        print(f"Top singular values: {sigma[:min(5,len(sigma))]}")
        print(f"Frobenius error ||A - A_k||_F = {err:.4f}")
        rel_err = err / np.linalg.norm(A, 'fro')
        print(f"Relative error = {rel_err:.4f}")

        # Save reconstructed image in same format
        save_same_format(Ak, input_path, k)
        results.append((k, Ak, err))

    # === Display all images ===
    ncols = len(results) + 1
    plt.figure(figsize=(3*ncols, 4))
    plt.subplot(1, ncols, 1)
    plt.imshow(A, cmap='gray', vmin=0, vmax=255)
    plt.title("Original")
    plt.axis('off')

    for i, (k, Ak, err) in enumerate(results):
        plt.subplot(1, ncols, i + 2)
        plt.imshow(np.clip(Ak, 0, 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        plt.title(f"k={k}\nErr={err:.1f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('my_plot2.png')
    plt.close()


if __name__ == "__main__":
    main()
