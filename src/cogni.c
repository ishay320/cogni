#include "cogni.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define ARR_LEN(arr) (sizeof(arr) / sizeof(arr[0]))
#define POW2(x) ((x) * (x))
#define UNUSED(var) (void)var
// ****************************
// ******* Math related *******
// ****************************

float mse(float x, float y)
{
    return POW2(x - y);
}

float mse_deriv(float truth, float pred)
{
    return -2 * (truth - pred);
}

float dot(const float* a, const float* b, size_t len)
{
    float sum = 0;
    for (size_t i = 0; i < len; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}
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
// ******* Neural related *******
// ******************************

float make_prediction(float* input_vector, float* weights, size_t len, float bias)
{
    float layer_1 = dot(input_vector, weights, len) + bias;
    float layer_2 = sigmoid(layer_1);
    return layer_2;
}

// output is: float derror_dbias, float* derror_dweights
void compute_gradients(float* derror_dbias, float derror_dweights[], const float* input_vector, const float* weights, const size_t len,
                       const float bias, const float target)
{
    // WIP
    float layer_1    = dot(input_vector, weights, len) + bias;
    float layer_2    = sigmoid(layer_1);
    float prediction = layer_2;

    float derror_dprediction  = 2 * (prediction - target);
    float dprediction_dlayer1 = sigmoid_deriv(layer_1);
    float dlayer1_dbias       = 1;
    float tmp1[len]; // can it work?!
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

typedef int error;

error writeWeights(const char* path, const float* weights, size_t w_len, const float* bias, size_t b_len)
{
    FILE* fp = fopen(path, "w");
    if (fp == 0)
    {
        fprintf(stderr, "could not open file '%s': %s", path, strerror(errno));
        return 1;
    }

    for (size_t i = 0; i < w_len; i++)
    {
        fprintf(fp, "%f ", weights[i]);
    }
    fprintf(fp, "\n");
    for (size_t i = 0; i < b_len; i++)
    {
        fprintf(fp, "%f ", bias[i]);
    }
    return 0;
}

error readWeights(const char* path, float* weights, size_t w_len, float* bias, size_t b_len)
{
    FILE* fp = fopen(path, "r");
    if (fp == 0)
    {
        fprintf(stderr, "could not open file '%s': %s", path, strerror(errno));
        return 1;
    }

    for (size_t i = 0; i < w_len; i++)
    {
        fscanf(fp, "%f ", &weights[i]);
    }
    fscanf(fp, "\n");
    for (size_t i = 0; i < b_len; i++)
    {
        fscanf(fp, "%f ", &bias[i]);
    }
    return 0;
}

typedef float (*activision)(float);
typedef struct Neuron
{
    activision fun;
    activision fun_derive;
    float* w;
    float* input;
    size_t w_len;
    float* b;
    float* out;
} Neuron;

typedef struct LinearLayer
{
    Neuron** n;
    size_t len;
    struct LinearLayer* last_layer;
} LinearLayer;

typedef struct Model
{
    activision loss_fun;
    activision loss_fun_derive;
    LinearLayer* l;
    size_t len;
} Model;

float linear(const float* w, const float* x, size_t len, float b)
{
    float sum = 0;
    for (size_t i = 0; i < len; i++)
    {
        sum += w[i] * x[i];
    }
    sum += b;
    return sum;
}

float linearForward(const Neuron* neuron)
{
    *neuron->out = neuron->fun(linear(neuron->w, neuron->input, neuron->w_len, *(neuron->b)));
    return *neuron->out;
}

// Setup the params
float xs[]     = {1.66, 1.56};
float y_true[] = {1};
float h[]      = {0, 0, 0};
float w[]      = {10.45, -10, 0, -3.9, 0.33, -4.7};
float b[]      = {3, 1, -5};

float lr = 0.9f;

int mine(int argc, char const* argv[])
{
    UNUSED(argc);
    UNUSED(argv);

    // read weights from file
    readWeights("simple", w, ARR_LEN(w), b, ARR_LEN(b));

    // setup of neurons
    Neuron h1 = {.fun = sigmoid, .fun_derive = sigmoid_deriv, .w = w + 0, .input = xs, .w_len = 2, .b = b + 0, .out = h + 0};
    Neuron h2 = {.fun = sigmoid, .fun_derive = sigmoid_deriv, .w = w + 2, .input = xs, .w_len = 2, .b = b + 1, .out = h + 1};
    Neuron o1 = {.fun = sigmoid, .fun_derive = sigmoid_deriv, .w = w + 4, .input = h, .w_len = 2, .b = b + 2, .out = h + 2};

    // forward pass (prediction)
    linearForward(&h1);
    linearForward(&h2);
    linearForward(&o1);
    printf("prediction: %f, truth: %f, mse: %f\n", h[2], y_true[0], mse(h[2], y_true[0]));

    // backpropagation (training)
    // TODO: Implement this

    return 0;
}

int reference(int argc, char const* argv[])
{
    UNUSED(argc);
    UNUSED(argv);

    readWeights("simple", w, ARR_LEN(w), b, ARR_LEN(b));

    // forward pass (prediction)
    float n1 = xs[0] * w[0] + xs[1] * w[1] + b[0];
    float h1 = sigmoid(n1);
    float n2 = xs[0] * w[2] + xs[1] * w[3] + b[1];
    float h2 = sigmoid(n2);
    float o1 = h1 * w[4] + h2 * w[5] + b[2];
    float a1 = sigmoid(o1);

    const float prediction = a1;
    printf("prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0], mse(prediction, y_true[0]));

    // backpropagation (training)
#if 1
    const size_t epochs = 20;

    float d_w[6] = {0};
    float d_b[3] = {0};
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float d_mse = mse_deriv(y_true[0], prediction);
        // o1 derivatives
        {
            const float base = sigmoid_deriv(a1) * d_mse;
            d_w[4]           = h1 * base;
            d_w[5]           = h2 * base;
            d_b[2]           = 1 * base;
        }

        // derivatives by h1 and h2
        float d_w_4 = w[4] * sigmoid_deriv(a1) * d_mse;
        float d_w_5 = w[5] * sigmoid_deriv(a1) * d_mse;

        // h1 derivatives
        {
            const float base = sigmoid_deriv(h1) * d_w_4;
            d_w[0]           = xs[0] * base;
            d_w[1]           = xs[1] * base;
            d_b[0]           = 1 * base;
        }
        // h2 derivatives
        {
            const float base = sigmoid_deriv(h2) * d_w_5;
            d_w[2]           = xs[0] * base;
            d_w[3]           = xs[1] * base;
            d_b[1]           = 1 * base;
        }

        // Applying the derivatives
        for (size_t i = 0; i < ARR_LEN(w); i++)
        {
            w[i] -= lr * d_w[i];
        }
        for (size_t i = 0; i < ARR_LEN(b); i++)
        {
            b[i] -= lr * d_b[i];
        }

        // forward pass (prediction)
        float n1 = xs[0] * w[0] + xs[1] * w[1] + b[0];
        float h1 = sigmoid(n1);
        float n2 = xs[0] * w[2] + xs[1] * w[3] + b[1];
        float h2 = sigmoid(n2);
        float o1 = h1 * w[4] + h2 * w[5] + b[2];
        float a1 = sigmoid(o1);

        const float prediction = a1;
        printf("prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0], mse(prediction, y_true[0]));

        // printArr(w, ARR_LEN(w), "weights");
        // printArr(b, ARR_LEN(b), "bias");
        // printArr(d_w, ARR_LEN(d_w), "d_weights");
        // printArr(d_b, ARR_LEN(d_b), "d_bias");
    }
    printArr(w, ARR_LEN(w), "weights");
    printArr(b, ARR_LEN(b), "bias");

    writeWeights("simple", w, ARR_LEN(w), b, ARR_LEN(b));
#endif
    return 0;
}

int main(int argc, char const* argv[])
{
    mine(argc, argv);
    printf("mine^ reference v\n");
    reference(argc, argv);
    return 0;
}
