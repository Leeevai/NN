#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#define NN_ASSERT assert
#endif

#pragma region Matrix.h
typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;

} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]  
#define MAT_PRINT(m) mat_print(m, #m, 0)
#define ARRAY_LEN(x) sizeof((x)) / sizeof((x)[0])

float rand_float(void);
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_print(Mat m, const char *name, size_t padding);
void mat_fill(Mat m, float a);
void mat_sig(Mat m);

#pragma endregion

#pragma region NN.h
typedef struct
{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // +1

} NN;

#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_RAND(nn) nn_rand((nn), 0, 1)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_zero(NN nn);

#pragma endregion

#endif










#ifdef NN_IMPLEMENTATION



#pragma region additional
float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}
#pragma endregion



#pragma region MATRIX
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(rows * cols * sizeof(*m.es));
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(dst.rows == a.rows && dst.cols == b.cols);
    NN_ASSERT(a.cols == b.rows);
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < b.cols; j++)
        {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.cols == a.cols);
    NN_ASSERT(dst.rows == a.rows);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_fill(Mat m, float a)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = a;
        }
    }
}

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s: ", (int)padding, "", name);
    printf("[\n");
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s", (int)padding, " ");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("     %f", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n\n", (int)padding, "");
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat){.rows = 1, .cols = m.cols, .stride = m.stride, .es = &MAT_AT(m, row, 0)};
}
void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows && dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    NN_ASSERT(low <= high);
    for (size_t i = 0; i < m.rows; i++)
    {

        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {

        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}
#pragma endregion



#pragma region NN

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    nn.as = NN_MALLOC(sizeof(*nn.as) * (1 + nn.count));
    NN_ASSERT(nn.ws != NULL && nn.bs != NULL && nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++)
    {
        nn.ws[i - 1] = mat_alloc(arch[i - 1], arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}
void nn_print(NN nn, const char *name)
{
    char buff[256];
    printf("%s: ", name);
    printf("[\n");
    for (size_t i = 0; i < nn.count; i++)
    {
        printf("\n\nlayer %zu\n", i);
        snprintf(buff, sizeof(buff), "ws[%zu]", i);
        mat_print(nn.ws[i], buff, 6);
        snprintf(buff, sizeof(buff), "bs[%zu]", i);
        mat_print(nn.bs[i], buff, 6);
    }
    printf("]\n");
}
void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}
void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}
float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == nn.as[nn.count].cols);
    float c = 0;

    for (size_t i = 0; i < ti.rows; ++i)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        for (size_t j = 0; j < to.cols; ++j)
        {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
            
        }
    }
    return c / (float)ti.rows;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
    float saved;
    float c = nn_cost(nn, ti, to);
    for (size_t i = 0; i < nn.count; i++)
    {

        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_zero(NN nn)
{
    for(size_t i = 0; i < nn.count; i++)
    {
        mat_fill(nn.bs[i],0);
        mat_fill(nn.as[i],0);
        mat_fill(nn.ws[i],0);
        if(i+1==nn.count)
        {
            mat_fill(nn.as[i+1],0);
        }
    }
}
void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i) {
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; ++j) {
            mat_fill(g.as[j], 0);
        }

        for (size_t j = 0; j < to.cols; ++j) {
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        for (size_t l = nn.count; l > 0; --l) {
            for (size_t j = 0; j < nn.as[l].cols; ++j) {
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                MAT_AT(g.bs[l-1], 0, j) += 2*da*a*(1 - a);
                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                    // j - weight matrix col
                    // k - weight matrix row
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += 2*da*a*(1 - a)*pa;
                    MAT_AT(g.as[l-1], 0, k) += 2*da*a*(1 - a)*w;
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; ++i) {
        for (size_t j = 0; j < g.ws[i].rows; ++j) {
            for (size_t k = 0; k < g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < g.bs[i].rows; ++j) {
            for (size_t k = 0; k < g.bs[i].cols; ++k) {
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
} 

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

#pragma endregion

#endif