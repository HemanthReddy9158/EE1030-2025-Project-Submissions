Project Title : Truncated SVD Image Compression

🛠️ Setup and Build Instructions

Prerequisites:
1.Ensure you have the following installed on your system (Ubuntu/Linux)
  Compiler         : clang (or gcc)
  Python Libraries : numpy, pillow, and matplotlib.


2.Place your target grayscale image (e.g., globe.jpg) inside the directory, as the Python script is configured to look there.

3.Build the C Backend:

The C source file must be compiled into a shared library (svd_lib.so) and placed in the folder for the Python driver to load it.

Navigate to the project directory and run the following compilation command:

clang -shared -o svd_lib.so -fPIC svd_lib.c -lm

NOTE:If svd_lib.so file is in another directory you have to change the path in the command accordingly.

    Ex: clang -shared -o codes/hybrid_c_python/c_backend/svd_lib.so \-fPIC codes/hybrid_c_python/c_backend/svd_lib.c -lm

🚀 Run Instructions

After successful compilation svd_lib.so file is created.

Now run python file using command

python3 (filename).py

The python script performs the following actions:

    Computes truncated SVD for k={5,20,50,100}.

    Prints the Frobenius Norm Error (∥A−Ak∥F) and the Compression Ratio to the terminal for each k.

    Saves the reconstructed images (e.g., globe_recon_k5.jpg).

    Saves a composite plot showing the original and all reconstructed images.
