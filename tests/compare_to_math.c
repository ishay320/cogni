#include "cogni.h"

#include <string.h>

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
const size_t epochs = 5;

int mine(char* out)
{
    readWeights("simple", w, ARR_LEN(w), b, ARR_LEN(b));

    Neuron* h1 = neuronInit(xs, w + 0, b + 0, dw + 0, db + 0, 2, sigmoid, sigmoidDeriv, h + 0);
    Neuron* h2 = neuronInit(xs, w + 2, b + 1, dw + 2, db + 1, 2, sigmoid, sigmoidDeriv, h + 1);
    Neuron* a1 = neuronInit(h, w + 4, b + 2, dw + 4, db + 2, 2, sigmoid, sigmoidDeriv, h + 2);

    // forward pass (prediction)
    neuronForward(h1);
    neuronForward(h2);
    neuronForward(a1);
    sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", h[2], y_true[0],
            mse(h[2], y_true[0]));

    // backpropagation (training)

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
        sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", h[2], y_true[0],
                mse(h[2], y_true[0]));
    }

    neuronDestroy(h1);
    neuronDestroy(h2);
    neuronDestroy(a1);
    return 0;
}

int reference(char* out)
{
    readWeights("simple", w, ARR_LEN(w), b, ARR_LEN(b));

    // forward pass (prediction)
    float n1 = xs[0] * w[0] + xs[1] * w[1] + b[0];
    float h1 = sigmoid(n1);
    float n2 = xs[0] * w[2] + xs[1] * w[3] + b[1];
    float h2 = sigmoid(n2);
    float o1 = h1 * w[4] + h2 * w[5] + b[2];
    float a1 = sigmoid(o1);

    const float prediction = a1;
    sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0],
            mse(prediction, y_true[0]));

    // backpropagation (training)

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
        sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0],
                mse(prediction, y_true[0]));
    }
    return 0;
}

int main(void)
{
    char my_lib[512]    = {0};
    char pure_math[512] = {0};
    mine(my_lib);
    reference(pure_math);

    int cmp = strcmp(my_lib, pure_math);
    if (cmp != 0)
    {
        printf(
            "\033[31m[-] %s test failed: my lib and the pure math output are not the same:\033[0m\n"
            "my lib\n%spure math:\n%s\n",
            __FILE__, my_lib, pure_math);
    }
    else
    {
        printf("\033[32m[+] %s test passed\033[0m\n", __FILE__);
    }

    return 0;
}
