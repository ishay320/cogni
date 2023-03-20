#include "cogni.h"

#include <math.h>
#include <stdio.h>

#define ARR_LEN(arr) (sizeof(arr) / sizeof(arr[0]))
#define POW2(x) ((x) * (x))

typedef struct
{
    float* weights;
    size_t len;
    float bias;
} Layer;

float mse(float x, float y) { return POW2(x - y); }

float mse_deriv(float truth, float pred) { return -2 * (truth - pred); }

float dot(const float* a, const float* b, size_t len)
{
    float sum = 0;
    for (size_t i = 0; i < len; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
float sigmoid_deriv(float x)
{
    float sig = sigmoid(x);
    return sig * (1.f - sig);
}

float* vector_add(float* a, const float* b, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        a[i] += b[i];
    }
    return a;
}

float* vector_add_to(float* out, const float* a, const float* b, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        out[i] = a[i] + b[i];
    }
    return out;
}

float* vector_mult_scalar(float* vec, size_t len, float scalar)
{
    for (size_t i = 0; i < len; i++)
    {
        vec[i] *= scalar;
    }
    return vec;
}

float* vector_mult_scalar_to(float* out, const float* vec, size_t len, float scalar)
{
    for (size_t i = 0; i < len; i++)
    {
        out[i] = vec[i] * scalar;
    }
    return out;
}

// ******************************

float make_prediction(float* input_vector, float* weights, size_t len, float bias)
{
    float layer_1 = dot(input_vector, weights, len) + bias;
    float layer_2 = sigmoid(layer_1);
    return layer_2;
}

// output is: float derror_dbias, float* derror_dweights
void compute_gradients(float* derror_dbias, float derror_dweights[], const float* input_vector, const float* weights, const size_t len, const float bias, const float target)
{
    // WIP
    float layer_1    = dot(input_vector, weights, len) + bias;
    float layer_2    = sigmoid(layer_1);
    float prediction = layer_2;

    float derror_dprediction  = 2 * (prediction - target);
    float dprediction_dlayer1 = sigmoid_deriv(layer_1);
    float dlayer1_dbias       = 1;
    float tmp1[len];  // can it work?!
    float tmp2[len];
    float* dlayer1_dweights = vector_add(vector_mult_scalar_to(tmp2, weights, len, 0), vector_mult_scalar_to(tmp1, input_vector, len, 1), len);

    *derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias);
    vector_mult_scalar_to(derror_dweights, dlayer1_dweights, len, derror_dprediction * dprediction_dlayer1);
}

void update_parameters(float* weights, float bias, float derror_dbias, float* derror_dweights, const size_t len, const float learning_rate)
{
    bias = bias - (derror_dbias * learning_rate);

    float* we_err = vector_mult_scalar(derror_dweights, len, learning_rate);
    for (size_t i = 0; i < len; i++)
    {
        weights[i] -= we_err[i];
    }
}

void printArr(float* arr, size_t len, char* name)
{
    printf("%s: ", name);
    for (size_t j = 0; j < len; j++)
    {
        printf("%f,", arr[j]);
    }
    printf("\n");
}

float xs[]     = {1.66, 1.56};
float y_true[] = {1};
float w[]      = {10.45, -10, 0, -3.9, 0.33, -4.7};
float b[]      = {3, 1, -5};

float lr = 0.9f;

int main(int argc, char const* argv[])
{
    // forward pass (prediction)

    float h1   = xs[0] * w[0] + xs[1] * w[1] + b[0];
    float h1_s = sigmoid(h1);
    float h2   = xs[0] * w[2] + xs[1] * w[3] + b[1];
    float h2_s = sigmoid(h2);
    float o1   = h1_s * w[4] + h2_s * w[5] + b[2];
    float o1_s = sigmoid(o1);

    float prediction = o1_s;
    printf("prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0], mse(prediction, y_true[0]));

    // backpropogation (training)

    float d_w[6] = {0};
    float d_b[3] = {0};
    for (size_t i = 0; i < 1000; i++)
    {
        // global
        float d_mse = mse_deriv(y_true[0], prediction);
        // o1 derivatives
        {
            d_w[4] = h1 * sigmoid_deriv(o1) * d_mse;
            d_w[5] = h2 * sigmoid_deriv(o1) * d_mse;
            d_b[2] = 1 * sigmoid_deriv(o1) * d_mse;
        }

        float d_w_4 = w[4] * sigmoid_deriv(o1) * d_mse;
        float d_w_5 = w[5] * sigmoid_deriv(o1) * d_mse;
        float d_b_1 = 1 * sigmoid_deriv(o1) * d_mse;

        // h1 derivatives
        {
            d_w[0] = xs[0] * sigmoid_deriv(h1) * d_w_4;
            d_w[1] = xs[1] * sigmoid_deriv(h1) * d_w_5;
            d_b[0] = 1 * sigmoid_deriv(h1) * d_b_1;
        }
        // h2 derivatives
        {
            d_w[2] = xs[0] * sigmoid_deriv(h2) * d_w[4];
            d_w[3] = xs[1] * sigmoid_deriv(h2) * d_w[5];
            d_b[1] = 1 * sigmoid_deriv(h2) * d_b[2];
        }

        for (size_t i = 0; i < ARR_LEN(w); i++)
        {
            w[i] -= lr * d_w[i];
        }
        for (size_t i = 0; i < ARR_LEN(b); i++)
        {
            b[i] -= lr * d_b[i];
        }

        // forward pass (prediction)

        h1   = xs[0] * w[0] + xs[1] * w[1] + b[0];
        h1_s = sigmoid(h1);
        h2   = xs[0] * w[2] + xs[1] * w[3] + b[1];
        h2_s = sigmoid(h2);
        o1   = h1_s * w[4] + h2_s * w[5] + b[2];
        o1_s = sigmoid(o1);

        prediction = o1_s;
        printf("prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0], mse(prediction, y_true[0]));

        // printArr(w, ARR_LEN(w), "weights");
        // printArr(b, ARR_LEN(b), "bias");
        // printArr(d_w, ARR_LEN(d_w), "d_weights");
        // printArr(d_b, ARR_LEN(d_b), "d_bias");
    }
    printArr(w, ARR_LEN(w), "weights");
    printArr(b, ARR_LEN(b), "bias");

    return 0;
}

// https://victorzhou.com/blog/intro-to-neural-networks/