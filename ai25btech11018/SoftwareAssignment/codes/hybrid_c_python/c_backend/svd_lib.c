#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

static double dot_vec(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static void matmul_AtA(const double* A, int m, int n, double* S) {
    // S = A^T * A  (n x n)
    int nn = n * n;
    for (int i = 0; i < nn; ++i) S[i] = 0.0;

    for (int r = 0; r < m; ++r) {
        const double* row = A + r * n;
        for (int i = 0; i < n; ++i) {
            double ri = row[i];
            if (ri == 0.0) continue;
            for (int j = 0; j < n; ++j) {
                S[i * n + j] += ri * row[j];
            }
        }
    }
}

static void mat_vec(const double* M, int rows, int cols, const double* x, double* y) {
    // y = M * x   (rows x cols) * (cols) -> (rows)
    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        const double* row = M + i * cols;
        for (int j = 0; j < cols; ++j) s += row[j] * x[j];
        y[i] = s;
    }
}

static void sym_mat_vec_minus_projections(const double* S, int n,
    const double* x, double* y,
    const double* prev_vs, const double* prev_lams, int found) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double* row = S + i * n;
        for (int j = 0; j < n; ++j) s += row[j] * x[j];
        y[i] = s;
    }
    // subtract deflation
    for (int t = 0; t < found; ++t) {
        const double* v = prev_vs + t * n;
        double lam = prev_lams[t];
        double dotvx = 0.0;
        for (int i = 0; i < n; ++i) dotvx += v[i] * x[i];
        double coeff = lam * dotvx;
        for (int i = 0; i < n; ++i) y[i] -= coeff * v[i];
    }
}

void truncated_svd(const double* A, int m, int n, int k, double* Ak_out, double* sigma_out) {
    if (k <= 0 || k > n) return;
    // 1) compute S = A^T A  (n x n)
    double* S = (double*) malloc(sizeof(double) * n * n);
    matmul_AtA(A, m, n, S);

    // allocate storage for eigenvectors (V) and eigenvalues (Lambda)
    double* V = (double*) malloc(sizeof(double) * n * k); // each column is an eigenvector of S
    double* Lambda = (double*) malloc(sizeof(double) * k);

    double* b = (double*) malloc(sizeof(double) * n);
    double* y = (double*) malloc(sizeof(double) * n);
    double* tmp = (double*) malloc(sizeof(double) * n);

    for (int i = 0; i < n; ++i) b[i] = 1.0 / (n>0? (double)n : 1.0);

    // For each top eigenpair, do power iteration with orthogonal deflation (by subtracting found components)
    for (int comp = 0; comp < k; ++comp) {
        // initialize b
        for (int i = 0; i < n; ++i) b[i] = (0.1 + (i+1.0)/ (n+1.0));

        double lambda = 0.0;
        for (int iter = 0; iter < 2000; ++iter) {
            // y = S*b  (and subtract deflation)
            sym_mat_vec_minus_projections(S, n, b, y, V, Lambda, comp);

            // normalize y
            double normy = 0.0;
            for (int i = 0; i < n; ++i) normy += y[i] * y[i];
            normy = sqrt(normy);
            if (normy == 0.0) break;
            for (int i = 0; i < n; ++i) tmp[i] = y[i] / normy;

            // Eigenvalue estimation
            sym_mat_vec_minus_projections(S, n, tmp, y, V, Lambda, comp);
            double new_lambda = dot_vec(tmp, y, n);

            // check convergence
            if (iter > 0 && fabs(new_lambda - lambda) < 1e-9 * fabs(new_lambda + 1e-12)) {
                lambda = new_lambda;
                // copy tmp into b (converged eigenvector)
                for (int i = 0; i < n; ++i) b[i] = tmp[i];
                break;
            }

            lambda = new_lambda;
            for (int i = 0; i < n; ++i) b[i] = tmp[i];
        }

        // store eigenvector and eigenvalue
        for (int i = 0; i < n; ++i) V[comp * n + i] = b[i];
        Lambda[comp] = lambda;
        // continue to next component
    }

    // Now compute singular values sigma_j = sqrt(max(lambda,0))
    double* sigma = (double*) malloc(sizeof(double) * k);
    for (int j = 0; j < k; ++j) {
        double lam = Lambda[j];
        if (lam < 0 && lam > -1e-12) lam = 0.0;
        sigma[j] = (lam > 0.0) ? sqrt(lam) : 0.0;
        sigma_out[j] = sigma[j];
    }

    // Compute left singular vectors U_j
    double* U = (double*) malloc(sizeof(double) * m * k);
    for (int j = 0; j < k; ++j) {
        const double* vj = V + j * n;
        // tmp (length m) = A * vj
        for (int i = 0; i < m; ++i) {
            double s = 0.0;
            const double* row = A + i * n;
            for (int t = 0; t < n; ++t) s += row[t] * vj[t];
            tmp[i] = s;
        }
        double sj = sigma[j];
        if (sj > 1e-12) {
            for (int i = 0; i < m; ++i) U[j * m + i] = tmp[i] / sj;
        } else {
            
            for (int i = 0; i < m; ++i) U[j * m + i] = 0.0;
        }
    }

    // Reconstruct Ak_out = sum{j=0..k-1} sigma_j * u_j * v_j^T
    // initialize Ak_out to 0
    for (int i = 0; i < m * n; ++i) Ak_out[i] = 0.0;

    for (int j = 0; j < k; ++j) {
        double sj = sigma[j];
        const double* uj = U + j * m; // length m
        const double* vj = V + j * n; // length n
        if (sj == 0.0) continue;
        for (int r = 0; r < m; ++r) {
            double ur = uj[r];
            double *rowA = Ak_out + r * n;
            for (int c = 0; c < n; ++c) {
                rowA[c] += sj * ur * vj[c];
            }
        }
    }

    free(S);
    free(V);
    free(Lambda);
    free(b);
    free(y);
    free(tmp);
    free(sigma);
    free(U);
}

