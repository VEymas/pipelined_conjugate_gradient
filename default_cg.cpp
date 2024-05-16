#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

namespace {
    const double EPS = std::numeric_limits<float>::epsilon();
    const int ione = 1;
    const double dnone = -1;
    const double done = 1;
}

extern "C" { // Scales a vector by a constant
    void dscal_(const int *, const double *, double *, const int *);
}

extern "C" { // Vector copy
    void dcopy_(const int *, const double *, const int *, double *, const int *);
}

extern "C" { // Dot product
    double ddot_(const int *, const double *, const int *, const double *, const int *);
}

extern "C" { // Vector norm
    double dnrm2_(const int *, const double *, const int *);
}

extern "C" { // y := a*x + y
    void daxpy_(const int *, const double *, const double *, const int *, double *, const int *);
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

        void matvec(double *vec, double *res_vec) {
            for (int i = 0; i < rows.size(); ++i) {
                int prev = (i == 0) ? 0 : rows[i - 1];
                double cur = 0;
                for (int j = prev; j < rows[i]; ++j) {
                    cur += values[j] * vec[columns[j]];
                } 
                res_vec[i] = cur;
            }
        }

        int get_rows_size() {
            return rows.size();
        }

    private:
        vector<int> rows;
        vector<int> columns;
        vector<double> values;
};

void print_vec(int size, double* vec) {
    for (int i = 0; i < size; ++i) {
        cout << vec[i] << " ";
    }
    cout << endl;
}

void PCG(Matrix_CSR& A, Matrix_CSR& M, double *b, double *x, int max_iter) {
    int size = A.get_rows_size();
    double *r = new double[size];
    double *u = new double[size];
    double *p = new double[size];
    double *s = new double[size];
    double gamma;
    double eta;
    double beta;

    double *matvec_res = new double[size];
    A.matvec(x, matvec_res);
    dcopy_(&size, b, &ione, r, &ione); // r := b
    daxpy_(&size, &dnone, matvec_res, &ione, r, &ione); // r := b - A * x
    free(matvec_res);

    M.matvec(r, u); // u := M * r
    dcopy_(&size, u, &ione, p, &ione); //p := u
    A.matvec(p, s); // s := A * p
    gamma = ddot_(&size, u, &ione, r, &ione); // gamma := (u, r)
    eta = ddot_(&size, s, &ione, p, &ione); // eta := (s, p)
    
    for (int i = 0; i < max_iter; ++i) {
        double alpha = gamma / eta;
        daxpy_(&size, &alpha, p, &ione, x, &ione); // x:= x + alpha * p
        double neg_alpha = -alpha;
        daxpy_(&size, &neg_alpha, s, &ione, r, &ione); // r:= r + alpha * s
        if (dnrm2_(&size, r, &ione) < 0.1) {
            break;
        }
        M.matvec(r, u); // u := M * r
        double gamma_prev = gamma;
        gamma = ddot_(&size, u, &ione, r, &ione); // gamma := (u, r)
        beta = gamma / gamma_prev;
        dscal_(&size, &beta, p, &ione); // p := betta * p
        daxpy_(&size, &done, u, &ione, p, &ione); // p := u + betta * p
        A.matvec(p, s); // s := A * p
        eta = ddot_(&size, s, &ione, p, &ione); // eta := (s, p)
    }
    free(r);
    free(u);
    free(p);
    free(s);
}

int main() {
    cout << "This is prog with blas" << endl;
    int size = 100;
    srand(time(0));
    vector<vector<double>> matrix_(size, vector<double>(size, 0));
    vector<vector<double>> M_(size, vector<double>(size, 0));
    for (int i = 0; i < size; ++i) {
        matrix_[i][i] = 4;
        M_[i][i] = 1;
        if (i + 1 < size) {
            matrix_[i + 1][i] = 1;
            matrix_[i][i + 1] = 1;
        }
    } 
    Matrix_CSR matrix(matrix_);
    Matrix_CSR M(M_);
    double *x = new double [size];
    double *b = new double [size];
    for (int i = 0; i < size; ++i) {
        x[i] =  rand();
        // x[i] = i;
    }
    matrix.matvec(x, b);
    double *my_x = new double [size];
    for (int i = 0; i < size; ++i) {
        my_x[i] = 0;
    }
    PCG(matrix, M, b, my_x, 100000);
    // for (int i = 0; i < size; ++i) {
    //     cout << x[i] << " ";
    // }
    // cout << endl;
    // for (int i = 0; i < size; ++i) {
    //     cout << my_x[i] << " ";
    // }
    // cout << endl;
    for (int i = 0; i < size; ++i) {
        x[i] -= my_x[i];
    }
    cout << dnrm2_(&size, x, &ione) << endl;
}