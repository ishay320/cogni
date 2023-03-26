#include "cogni.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

float mseDeriv(float truth, float pred)
{
    return -2 * (truth - pred);
}

float vector_dot(const float* a, const float* b, size_t len)
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

float sigmoidDeriv(float x)
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

error writeWeights(const char* path, const float* weights, size_t w_len, const float* bias,
                   size_t b_len)
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
/* the data that will be provided:
    float x[];  // input
    float w[];  // weights
    float b[];  // biases
    float dw[]; // weights derivatives
    float db[]; // bias derivatives

    the data that is not provided:
    intermidiate derivatives
    the neurons list


! what about neuron output? - its the responsibility of layer but its also needed in calculating
                              grads - also partial gradients
 */

typedef float (*activision)(float);

typedef struct _Neuron
{
    activision fun;
    activision fun_derive;

    // weights bias and inputs
    float* w;
    float* x;
    size_t w_len;
    float* b;

    // derivatives
    float base_derive;
    float* dw;
    float* db;

    // outputs
    float* out;
} Neuron;

Neuron* neuronInit(float x[], float w[], float* b, float dw[], float* db, size_t len,
                   activision fun, activision fun_derive, float* out)
{
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (neuron == 0)
    {
        fprintf(stderr, "[ERROR] Could not allocate memory for neuron.\n");
        exit(EXIT_FAILURE);
    }

    neuron->fun        = fun;
    neuron->fun_derive = fun_derive;

    neuron->w     = w;
    neuron->x     = x;
    neuron->w_len = len;
    neuron->b     = b;
    neuron->dw    = dw;
    neuron->db    = db;

    neuron->base_derive = 0;

    neuron->out = out;

    return neuron;
}

void neuronDestroy(Neuron* neuron)
{
    free(neuron);
}

typedef struct LinearLayer
{
    Neuron** n;
    float* outputs;
    float* part_derive;
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

float neuronForward(Neuron* neuron)
{
    const float linear_out = linear(neuron->w, neuron->x, neuron->w_len, *(neuron->b));
    const float activated  = neuron->fun(linear_out);
    *neuron->out           = activated;
    return *neuron->out;
}

void neuronBackPropagate(Neuron* neuron, float part_derive)
{
    neuron->base_derive = neuron->fun_derive(*neuron->out) * part_derive;
    for (size_t i = 0; i < neuron->w_len; i++)
    {
        neuron->dw[i] = neuron->base_derive * neuron->x[i];
    }
    *neuron->db = neuron->base_derive;
}

void neuronCalculatePartDerive(Neuron* neuron, float* part_derives)
{
    for (size_t i = 0; i < neuron->w_len; i++)
    {
        part_derives[i] = neuron->base_derive * neuron->w[i];
    }
}

void applyDerives(float* w, float* dw, size_t w_len, float* b, float* db, size_t b_len, float lr)
{
    for (size_t i = 0; i < w_len; i++)
    {
        w[i] -= lr * dw[i];
    }

    for (size_t i = 0; i < b_len; i++)
    {
        b[i] -= lr * db[i];
    }
}

#define INPUTS_LEN 2
#define NEURONS_LEN 3
#define WEIGHTS_LEN (NEURONS_LEN * INPUTS_LEN)
#define BIAS_LEN (WEIGHTS_LEN / INPUTS_LEN)
// Setup the params
float xs[INPUTS_LEN]  = {1.66, 1.56};
float y_true[]        = {1};
float h[NEURONS_LEN]  = {0, 0, 0};
float b[BIAS_LEN]     = {3, 1, -5};
float w[WEIGHTS_LEN]  = {10.45, -10, 0, -3.9, 0.33, -4.7};
float dw[WEIGHTS_LEN] = {0};
float db[BIAS_LEN]    = {0};

float lr            = 0.9f;
const size_t epochs = 2;

int mine(int argc, char const* argv[])
{
    UNUSED(argc);
    UNUSED(argv);

    readWeights("simple", w, ARR_LEN(w), b, ARR_LEN(b));

    Neuron* h1 = neuronInit(xs, w + 0, b + 0, dw + 0, db + 0, 2, sigmoid, sigmoidDeriv, h + 0);
    Neuron* h2 = neuronInit(xs, w + 2, b + 1, dw + 2, db + 1, 2, sigmoid, sigmoidDeriv, h + 1);
    Neuron* a1 = neuronInit(h, w + 4, b + 2, dw + 4, db + 2, 2, sigmoid, sigmoidDeriv, h + 2);

    // forward pass (prediction)
    neuronForward(h1);
    neuronForward(h2);
    neuronForward(a1);
    printf("prediction: %f, truth: %f, mse: %f\n", h[2], y_true[0], mse(h[2], y_true[0]));

// backpropagation (training)
#if 1

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float d_mse = mseDeriv(y_true[0], h[2]);

        neuronBackPropagate(a1, d_mse);

        float part_derive[2];
        neuronCalculatePartDerive(a1, part_derive);

        neuronBackPropagate(h1, part_derive[0]);
        neuronBackPropagate(h2, part_derive[1]);

        // Apply the derives
        applyDerives(w, dw, ARR_LEN(w), b, db, ARR_LEN(b), lr);

        // forward pass (prediction)
        neuronForward(h1);
        neuronForward(h2);
        neuronForward(a1);
        printf("prediction: %f, truth: %f, mse: %f\n", h[2], y_true[0], mse(h[2], y_true[0]));
    }
#endif

    neuronDestroy(h1);
    neuronDestroy(h2);
    neuronDestroy(a1);
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
    printf("prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0],
           mse(prediction, y_true[0]));

    // backpropagation (training)
#if 1

    float d_w[6] = {0};
    float d_b[3] = {0};
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float d_mse = mseDeriv(y_true[0], prediction);
        // o1 derivatives
        {
            const float base = sigmoidDeriv(a1) * d_mse;
            d_w[4]           = h1 * base;
            d_w[5]           = h2 * base;
            d_b[2]           = 1 * base;
        }

        // derivatives by h1 and h2
        float d_w_4 = w[4] * sigmoidDeriv(a1) * d_mse;
        float d_w_5 = w[5] * sigmoidDeriv(a1) * d_mse;

        // h1 derivatives
        {
            const float base = sigmoidDeriv(h1) * d_w_4;
            d_w[0]           = xs[0] * base;
            d_w[1]           = xs[1] * base;
            d_b[0]           = 1 * base;
        }
        // h2 derivatives
        {
            const float base = sigmoidDeriv(h2) * d_w_5;
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
        printf("prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0],
               mse(prediction, y_true[0]));

        // printArr(w, ARR_LEN(w), "weights");
        // printArr(b, ARR_LEN(b), "bias");
        // printArr(d_w, ARR_LEN(d_w), "d_weights");
        // printArr(d_b, ARR_LEN(d_b), "d_bias");
    }
    // printArr(w, ARR_LEN(w), "weights");
    // printArr(b, ARR_LEN(b), "bias");

    // writeWeights("simple", w, ARR_LEN(w), b, ARR_LEN(b));
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
