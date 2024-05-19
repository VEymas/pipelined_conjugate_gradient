#include <iostream>
#include <cuda_runtime.h>

__global__ void daxpy_kernel(const int n, const double a, const double *x, double *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}

void daxpy_gpu(int num_blocks, int block_size, int ncudaStream_t& stream, cudaEvent_t& event, const int size, const double a, const double *d_x, double *d_y) {
    daxpy_kernel<<<num_blocks, block_size, 0, stream>>>(size, a, d_x, d_y);
    cudaEventRecord(event, stream);

    cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}

__global__ void dpxay_kernel(const int n, const double a, const double *x, double *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = x[tid] + a * y[tid];
    }
}

void dpxay_gpu(int num_blocks, int block_size, int ncudaStream_t& stream, cudaEvent_t& event, const int size, const double a, const double *d_x, double *d_y) {
    daxpy_kernel<<<num_blocks, block_size, 0, stream>>>(size, a, d_x, d_y);
    cudaEventRecord(event, stream);

    cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}

__global__ void scal_vec_kernel(int size, double *a, double b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        a[tid] *= b;
    }
}

void scal_vec_gpu(int size, double *a, double b) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    double *d_a;
    cudaMalloc(&d_a, size * sizeof(double));
    cudaMemcpy(d_a, a, size * sizeof(double), cudaMemcpyHostToDevice);

    scal_vec_kernel<<<num_blocks, block_size, 0, stream>>>(size, d_a, b);

    cudaMemcpy(a, d_a, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
}

__global__ void dot_product_kernel(int size, double *a, double *b, double *result) {
    extern __shared__ double shared[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;

    shared[local_index] = (tid < size) ? a[tid] * b[tid] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (local_index < offset) {
            shared[local_index] += shared[local_index + offset];
        }
        __syncthreads();
    }

    if (local_index == 0) {
        result[blockIdx.x] = shared[0];
    }
}

void dot_product_gpu(int num_blocks, int block_size, cudaStream_t& stream, cudaEvent_t& event, int size, double *d_a, double *d_b) {
    dot_product_kernel<<<num_blocks, block_size, block_size * sizeof(double), stream>>>(size, d_a, d_b, d_result);
    cudaEventRecord(event, stream);
}

__global__ void vec_norm_kernel(int size, double *a, double *result) {
    extern __shared__ double shared[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;

    shared[local_index] = (tid < size) ? a[tid] * a[tid] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (local_index < offset) {
            shared[local_index] += shared[local_index + offset];
        }
        __syncthreads();
    }

    if (local_index == 0) {
        result[blockIdx.x] = shared[0];
    }
}

__global__ void matvec_kernel(int n, int *rows, int *columns, double *values, double *vec, double *res_vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int prev = (i == 0) ? 0 : rows[i - 1];
        double cur = 0;
        for (int j = prev; j < n; ++j) {
            cur += values[j] * vec[columns[j]];
        } 
        res_vec[i] = cur;
    }
}

void matvec_gpu(int num_blocks, int block_size, cudaStream_t& stream, cudaEvent_t& event, int size, int *d_rows, int *d_columns, double *d_values, double *d_vec, double *d_res_vec) {
    matvec_kernel<<<num_blocks, block_size, 0, stream>>>(size, d_rows, d_columns, d_values, d_vec, d_res_vec);
    cudaEventRecord(event, stream);
}

class Matrix_CSR { // CSR matrix format
    public:
        Matrix_CSR(vector<vector<double>> matrix) {
            int cnt = 0;
            for (int i = 0; i < matrix.size(); ++i) {
                for (int j = 0; j < matrix[0].size(); ++j) {
                    if (abs(matrix[i][j]) > EPS) {
                        values.emplace_back(matrix[i][j]);
                        columns.emplace_back(j);
                        ++cnt;
                    }
                }
                rows.emplace_back(cnt);
            }
        }

        vector<int> get_rows() {
            return rows;
        }

        vector<int> get_columns() {
            return columns;
        }

        vector<double> get_values(){
            return values;
        }

    private:
        vector<int> rows;
        vector<int> columns;
        vector<double> values;
};

int *arr_from_vec_int(vector<int>& vec) {
    double *res = new int[vec.size()];
    for (int i = 0; i < vec.size(); ++i) {
        res[i] = vec[i];
    }
    return res;
}

double *arr_from_vec_double(vector<double>& vec) {
    double *res = new double[vec.size()];
    for (int i = 0; i < vec.size(); ++i) {
        res[i] = vec[i];
    }
    return res;
}

void copy_vector(int size; const double *vec1, double vec2) {
    for (int i = 0; i < size; ++i) {
        vec2[i] = vec1[i];
    }
}

double vec_norm(int size, double *vec) {
    double res = 0;
    for (int i = 0; i < size; ++i) {
        res += vec[i];
    }
    return (sqrt(res));
}

void PCG(Matrix_CSR& A, Matrix_CSR& M, double *b, double *x, int max_iter) {
    int size = A.get_rows_size();
    double *r = new double[size];
    double *u = new double[size];
    double *w = new double[size];
    double *m = new double[size];
    double *n = new double[size];
    double *p = new double[size];
    double *s = new double[size];
    double *q = new double[size];
    double *z = new double[size];
    double gamma;
    double delta;
    double eta;
    double beta;
    double alpha;
    int *a_rows = arr_from_vec_int(A.get_rows());
    int *a_columns = arr_from_vec_int(A.get_columns());
    double *a_values = arr_from_vec_double(A.get_values());
    int *m_rows = arr_from_vec_int(M.get_rows());
    int *m_columns = arr_from_vec_int(M.get_columns());
    double *m_values = arr_from_vec_double(M.get_values());

    const int num_blocks = (size + block_size - 1) / block_size;
    const int block_size = 256;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    cudaStream_t stream3;
    cudaStreamCreate(&stream3);
    cudaStream_t stream4;
    cudaStreamCreate(&stream4);

    double *matvec_res = new double[size]; // matvec_res := Ax
    cudaStream_t tmp_stream;
    cudaStreamCreate(&tmp_stream);
    cudaEvent_t tmp_event;
    cudaEventCreate(&tmp_event);
    int *d_rows, *d_columns;
    double *d_values, *d_x, *d_res_vec;
    cudaMalloc(&d_rows, size * sizeof(int));
    cudaMalloc(&d_columns, size * sizeof(int));
    cudaMalloc(&d_values, size * sizeof(double));
    cudaMalloc(&d_vec, size * sizeof(double));
    cudaMalloc(&d_res_vec, size * sizeof(double));
    cudaMemcpy(d_rows, a_rows, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, a_columns, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, a_values, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    matvec_gpu(num_blocks, block_size, tmp_stream, tmp_event, size, d_rows, d_columns, d_values, d_x, d_res_vec);
    cudaEventSynchronize(tmp_event);
    cudaEventDestroy(tmp_event);
    cudaMemcpy(matvec_res, d_res_vec, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rows);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_vec);
    cudaFree(d_res_vec);

    copy_vector(size, b, r); // r:= b
    // r:= b - Ax
    double *d_x, *d_y;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_y, size * sizeof(double));
    cudaMemcpy(d_x, matvec_res, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, r, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaEvent_t r_event;
    cudaEventCreate(&r_event);
    daxpy_gpu(num_blocks, block_size, stream1, r_event, size, -1, d_x, d_y);
    cudaEventSynchronize(r_event);
    cudaMemcpy(y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    free(matvec_res);

    // u := Mr
    cudaEvent_t u_event;
    cudaEventCreate(&u_event);
    int *d_rows, *d_columns;
    double *d_values, *d_x, *d_res_vec;
    cudaMalloc(&d_rows, size * sizeof(int));
    cudaMalloc(&d_columns, size * sizeof(int));
    cudaMalloc(&d_values, size * sizeof(double));
    cudaMalloc(&d_vec, size * sizeof(double));
    cudaMalloc(&d_res_vec, size * sizeof(double));
    cudaMemcpy(d_rows, m_rows, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, m_columns, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, m_values, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, r, n * sizeof(double), cudaMemcpyHostToDevice);
    matvec_gpu(num_blocks, block_size, stream1, u_event, size, d_rows, d_columns, d_values, d_x, d_res_vec);
    cudaEventSynchronize(u_event);
    cudaMemcpy(u, d_res_vec, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rows);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_vec);
    cudaFree(d_res_vec);

    // w := Au
    cudaEvent_t w_event;
    cudaEventCreate(&w_event);
    int *d_rows, *d_columns;
    double *d_values, *d_x, *d_res_vec;
    cudaMalloc(&d_rows, size * sizeof(int));
    cudaMalloc(&d_columns, size * sizeof(int));
    cudaMalloc(&d_values, size * sizeof(double));
    cudaMalloc(&d_vec, size * sizeof(double));
    cudaMalloc(&d_res_vec, size * sizeof(double));
    cudaMemcpy(d_rows, a_rows, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, a_columns, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, a_values, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, u, size * sizeof(double), cudaMemcpyHostToDevice);
    matvec_gpu(num_blocks, block_size, stream1, w_event, size, d_rows, d_columns, d_values, d_x, d_res_vec);
    cudaEventSynchronize(w_event);
    cudaMemcpy(w, d_res_vec, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rows);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_vec);
    cudaFree(d_res_vec);

    // m := Mw
    cudaEvent_t m_event;
    cudaEventCreate(&m_event);
    int *d_rows, *d_columns;
    double *d_values, *d_x, *d_res_vec;
    cudaMalloc(&d_rows, size * sizeof(int));
    cudaMalloc(&d_columns, size * sizeof(int));
    cudaMalloc(&d_values, size * sizeof(double));
    cudaMalloc(&d_vec, size * sizeof(double));
    cudaMalloc(&d_res_vec, size * sizeof(double));
    cudaMemcpy(d_rows, m_rows, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, m_columns, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, m_values, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, w, size * sizeof(double), cudaMemcpyHostToDevice);
    matvec_gpu(num_blocks, block_size, stream1, m_event, size, d_rows, d_columns, d_values, d_x, d_res_vec);
    cudaEventSynchronize(m_event);
    cudaMemcpy(m, d_res_vec, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rows);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_vec);
    cudaFree(d_res_vec);

    // n:= Am
    cudaEvent_t n_event;
    cudaEventCreate(&n_event);
    int *d_rows, *d_columns;
    double *d_values, *d_x, *d_res_vec;
    cudaMalloc(&d_rows, size * sizeof(int));
    cudaMalloc(&d_columns, size * sizeof(int));
    cudaMalloc(&d_values, size * sizeof(double));
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_res_vec, size * sizeof(double));
    cudaMemcpy(d_rows, a_rows, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, a_columns, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, as_values, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, m, size * sizeof(double), cudaMemcpyHostToDevice);
    matvec_gpu(num_blocks, block_size, stream1, n_event, size, d_rows, d_columns, d_values, d_x, d_res_vec);
    cudaEventSynchronize(n_event);
    cudaMemcpy(n, d_res_vec, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_rows);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_vec);
    cudaFree(d_res_vec);

    // gamma := (u, r)
    cudaEvent_t g_event;
    cudaEventCreate(&g_event);
    double *d_a, *d_b, *d_result;
    double *h_result = new double[num_blocks];
    cudaMalloc(&d_a, size * sizeof(double));
    cudaMalloc(&d_b, size * sizeof(double));
    cudaMalloc(&d_result, num_blocks * sizeof(double));
    cudaMemcpy(d_a, u, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, r, size * sizeof(double), cudaMemcpyHostToDevice);
    dot_product_gpu(num_blocks, block_size, stream1, g_event, size, d_a, d_b);
    cudaEventSynchronize(g_event);
    cudaMemcpy(h_result, d_result, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double result = 0;
    for (int i = 0; i < num_blocks; ++i) {
        result += h_result[i];
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    delete[] h_result;
    gamma = result;


    // delta := (u, w)
    cudaEvent_t d_event;
    cudaEventCreate(&d_event);
    double *d_a, *d_b, *d_result;
    double *h_result = new double[num_blocks];
    cudaMalloc(&d_a, size * sizeof(double));
    cudaMalloc(&d_b, size * sizeof(double));
    cudaMalloc(&d_result, num_blocks * sizeof(double));
    cudaMemcpy(d_a, u, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, w, size * sizeof(double), cudaMemcpyHostToDevice);
    dot_product_gpu(num_blocks, block_size, stream1, d_event, size, d_a, d_b);
    cudaEventSynchronize(d_event);
    cudaMemcpy(h_result, d_result, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double result = 0;
    for (int i = 0; i < num_blocks; ++i) {
        result += h_result[i];
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    delete[] h_result;
    delta = result;

    eta = delta;
    copy_vector(size, u, p); //p := u;
    copy_vector(size, w, s); //s := w;
    copy_vector(size, m, q); //q := m;
    copy_vector(size, n, z); //z := n;
    alpha = gamma / eta;
    double malpha = -1 * alpha;

    for (int i = 0; i < max_iter; ++i) {
        // x := x + alpha*p
        double *d_p, *d_x;
        cudaMalloc(&d_x, size * sizeof(double));
        cudaMalloc(&d_p, n * sizeof(double));
        cudaMemcpy(d_p, p, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t x_event;
        cudaEventCreate(&x_event);
        daxpy_gpu(num_blocks, block_size, stream1, x_event, size, alpha, d_p, d_x);

        // r := r - alpha * s
        double *d_s, *d_r;
        cudaMalloc(&d_s, size * sizeof(double));
        cudaMalloc(&d_r, size * sizeof(double));
        cudaMemcpy(d_s, s, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t r_event;
        cudaEventCreate(&r_event);
        daxpy_gpu(num_blocks, block_size, stream2, r_event, size, malpha, d_s, d_r);

        // u := u - alpha * q
        double *d_q, *d_u;
        cudaMalloc(&d_q, size * sizeof(double));
        cudaMalloc(&d_u, size * sizeof(double));
        cudaMemcpy(d_q, q, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u, u, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t u_event;
        cudaEventCreate(&u_event);
        daxpy_gpu(num_blocks, block_size, stream3, u_event, size, malpha, d_q, d_u);

        // w := w - alpha * z
        double *d_z, *d_w;
        cudaMalloc(&d_q, size * sizeof(double));
        cudaMalloc(&d_u, size * sizeof(double));
        cudaMemcpy(d_z, z, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t w_event;
        cudaEventCreate(&w_event);
        daxpy_gpu(num_blocks, block_size, stream4, w_event, size, malpha, d_z, d_w);

        // Waiting w
        cudaEventSynchronize(w_event);
        cudaMemcpy(w, d_w, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_z);
        cudaFree(d_w);

        // m := M * w
        cudaEvent_t m_event;
        cudaEventCreate(&m_event);
        int *d_m_rows, *d_m_columns;
        double *d_m_values, *d_w, *d_m;
        cudaMalloc(&d_m_rows, size * sizeof(int));
        cudaMalloc(&d_m_columns, size * sizeof(int));
        cudaMalloc(&d_m_values, size * sizeof(double));
        cudaMalloc(&d_w, size * sizeof(double));
        cudaMalloc(&d_m, size * sizeof(double));
        cudaMemcpy(d_m_rows, m_rows, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m_columns, m_columns, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m_values, m_values, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w, size * sizeof(double), cudaMemcpyHostToDevice);
        matvec_gpu(num_blocks, block_size, stream4, m_event, size, d_rows, d_columns, d_values, d_x, d_res_vec);

        // Waiting u
        cudaEventSynchronize(u_event);
        cudaMemcpy(u, d_u, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_q);
        cudaFree(d_u);

        // delta := (u, w)
        cudaEvent_t d_event;
        cudaEventCreate(&d_event);
        double *d_u, *d_w, *d_d_result;
        double *h_d_result = new double[num_blocks];
        cudaMalloc(&d_u, size * sizeof(double));
        cudaMalloc(&d_w, size * sizeof(double));
        cudaMalloc(&d_d_result, num_blocks * sizeof(double));
        cudaMemcpy(d_u, u, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w, size * sizeof(double), cudaMemcpyHostToDevice);
        dot_product_gpu(num_blocks, block_size, stream3, d_event, size, d_u, d_w);

        // Waiting r
        cudaEventSynchronize(r_event);
        cudaMemcpy(r, d_r, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_s);
        cudaFree(d_r);

        if (vec_norm(size, r) < 0.1) {
            break;
        }

        double gamma_prev = gamma;
        // gamma := (u, r)
        cudaEvent_t g_event;
        cudaEventCreate(&g_event);
        double *d_g_u, *d_r, *d_g_result;
        double *h_g_result = new double[num_blocks];
        cudaMalloc(&d_g_u, size * sizeof(double));
        cudaMalloc(&d_r, size * sizeof(double));
        cudaMalloc(&d_g_result, num_blocks * sizeof(double));
        cudaMemcpy(d_g_u, u, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r, size * sizeof(double), cudaMemcpyHostToDevice);
        dot_product_gpu(num_blocks, block_size, stream2, g_event, size, d_g_u, d_r);

        // Waiting m
        cudaEventSynchronize(m_event);
        cudaMemcpy(m, d_m, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_m_rows);
        cudaFree(d_m_columns);
        cudaFree(d_m_values);
        cudaFree(d_w);
        cudaFree(d_m); 

        // n := Am
        cudaEvent_t n_event;
        cudaEventCreate(&n_event);
        int *d_a_rows, *d_a_columns;
        double *d_a_values, *d_m, *d_n;
        cudaMalloc(&d_a_rows, size * sizeof(int));
        cudaMalloc(&d_a_columns, size * sizeof(int));
        cudaMalloc(&d_a_values,size * sizeof(double));
        cudaMalloc(&d_m, size * sizeof(double));
        cudaMalloc(&d_n, size * sizeof(double));
        cudaMemcpy(d_a_rows, a_rows, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_columns, a_columns, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_values, as_values, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, m, size * sizeof(double), cudaMemcpyHostToDevice);
        matvec_gpu(num_blocks, block_size, stream4, n_event, size, d_a_rows, d_a_columns, d_a_values, d_m, d_n);

        // Waiting x
        cudaEventSynchronize(x_event);
        cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_p);
        cudaFree(d_x);

        // Waiting delta
        cudaEventSynchronize(d_event);
        cudaMemcpy(h_d_result, d_d_result, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
        double delta_result = 0;
        for (int i = 0; i < num_blocks; ++i) {
            delta_result += h_d_result[i];
        }
        cudaFree(d_u);
        cudaFree(d_w);
        cudaFree(d_d_result);
        delete[] h_d_result;
        delta = delta_result;

        // Waiting gamma
        cudaEventSynchronize(g_event);
        cudaMemcpy(h_g_result, d_g_result, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
        double gamma_result = 0;
        for (int i = 0; i < num_blocks; ++i) {
            gamma_result += h_result[i];
        }
        cudaFree(d_g_u);
        cudaFree(d_r);
        cudaFree(d_g_result);
        delete[] h_g_result;
        gamma = gamma_result;

        // Waiting n
        cudaEventSynchronize(n_event);
        cudaMemcpy(n, d_n, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_a_rows);
        cudaFree(d_a_columns);
        cudaFree(d_a_values);
        cudaFree(d_m);
        cudaFree(d_n);

        beta = gamma / gamma_prev;
        eta = delta - beta * beta * eta;
        alpha = gamma / eta;

        // p := u + beta * p
        double *d_u, *d_p;
        cudaMalloc(&d_p, size * sizeof(double));
        cudaMalloc(&d_u, size * sizeof(double));
        cudaMemcpy(d_u, u, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p, p, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t p_event;
        cudaEventCreate(&p_event);
        dpxay_gpu(num_blocks, block_size, stream1, p_event, size, beta, d_u, d_p);

        // s := w + beta * s
        double *d_w, *d_s;
        cudaMalloc(&d_w, size * sizeof(double));
        cudaMalloc(&d_s, size * sizeof(double));
        cudaMemcpy(d_w, w, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, s, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t s_event;
        cudaEventCreate(&s_event);
        dpxay_gpu(num_blocks, block_size, stream2, s_event, size, beta, d_w, d_s);

        // q := m + beta * q
        double *d_m, *d_q;
        cudaMalloc(&d_m, size * sizeof(double));
        cudaMalloc(&d_q, size * sizeof(double));
        cudaMemcpy(d_m, m, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, s, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t q_event;
        cudaEventCreate(&q_event);
        dpxay_gpu(num_blocks, block_size, stream3, q_event, size, beta, d_m, d_q);

        // z := n + beta * z
        double *d_n, *d_z;
        cudaMalloc(&d_n, size * sizeof(double));
        cudaMalloc(&d_z, size * sizeof(double));
        cudaMemcpy(d_n, n, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, z, size * sizeof(double), cudaMemcpyHostToDevice);
        cudaEvent_t z_event;
        cudaEventCreate(&z_event);
        dpxay_gpu(num_blocks, block_size, stream4, z_event, size, beta, d_n, d_z);

        // Waiting p
        cudaEventSynchronize(p_event);
        cudaMemcpy(p, d_p, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_u);
        cudaFree(d_p);

        // Waiting s
        cudaEventSynchronize(s_event);
        cudaMemcpy(s, d_s, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_w);
        cudaFree(d_s);

        // Waiting q
        cudaEventSynchronize(q_event);
        cudaMemcpy(q, d_q, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_m);
        cudaFree(d_q);

        // Waiting z
        cudaEventSynchronize(z_event);
        cudaMemcpy(z, d_z, size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_n);
        cudaFree(d_z);
    }
    
    free(r);
    free(u);
    free(w);
    free(m);
    free(n);
    free(p);
    free(s);
    free(q);
    free(z);
    free(a_rows);
    free(a_columns);
    free(a_values);
    free(m_rows);
    free(m_columns);
    free(m_values);
}