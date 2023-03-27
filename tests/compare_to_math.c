#define COGNI_IMPLEMENTATION
#include "cogni.h"

#include <string.h>

#define ARR_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

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
    cog_read_weights("simple", w, ARR_LEN(w), b, ARR_LEN(b));

    Neuron* h1 =
        cog_neuron_init(xs, w + 0, b + 0, dw + 0, db + 0, 2, cog_sigmoid, cog_sigmoid_deriv, h + 0);
    Neuron* h2 =
        cog_neuron_init(xs, w + 2, b + 1, dw + 2, db + 1, 2, cog_sigmoid, cog_sigmoid_deriv, h + 1);
    Neuron* a1 =
        cog_neuron_init(h, w + 4, b + 2, dw + 4, db + 2, 2, cog_sigmoid, cog_sigmoid_deriv, h + 2);

    // forward pass (prediction)
    cog_neuron_forward(h1);
    cog_neuron_forward(h2);
    cog_neuron_forward(a1);
    sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", h[2], y_true[0],
            cog_mse(h[2], y_true[0]));

    // backpropagation (training)

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float d_mse = cog_mse_deriv(y_true[0], h[2]);

        cog_neuron_backpropagate(a1, d_mse);

        float part_derive[2];
        cog_neuron_part_derive(a1, part_derive);

        cog_neuron_backpropagate(h1, part_derive[0]);
        cog_neuron_backpropagate(h2, part_derive[1]);

        // Apply the derives
        cog_apply_derives(w, dw, ARR_LEN(w), b, db, ARR_LEN(b), lr);

        // forward pass (prediction)
        cog_neuron_forward(h1);
        cog_neuron_forward(h2);
        cog_neuron_forward(a1);
        sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", h[2], y_true[0],
                cog_mse(h[2], y_true[0]));
    }

    cog_neuron_destroy(h1);
    cog_neuron_destroy(h2);
    cog_neuron_destroy(a1);
    return 0;
}

int reference(char* out)
{
    cog_read_weights("simple", w, ARR_LEN(w), b, ARR_LEN(b));

    // forward pass (prediction)
    float n1 = xs[0] * w[0] + xs[1] * w[1] + b[0];
    float h1 = cog_sigmoid(n1);
    float n2 = xs[0] * w[2] + xs[1] * w[3] + b[1];
    float h2 = cog_sigmoid(n2);
    float o1 = h1 * w[4] + h2 * w[5] + b[2];
    float a1 = cog_sigmoid(o1);

    const float prediction = a1;
    sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0],
            cog_mse(prediction, y_true[0]));

    // backpropagation (training)

    float d_w[6] = {0};
    float d_b[3] = {0};
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float d_mse = cog_mse_deriv(y_true[0], prediction);
        // o1 derivatives
        {
            const float base = cog_sigmoid_deriv(a1) * d_mse;
            d_w[4]           = h1 * base;
            d_w[5]           = h2 * base;
            d_b[2]           = 1 * base;
        }

        // derivatives by h1 and h2
        float d_w_4 = w[4] * cog_sigmoid_deriv(a1) * d_mse;
        float d_w_5 = w[5] * cog_sigmoid_deriv(a1) * d_mse;

        // h1 derivatives
        {
            const float base = cog_sigmoid_deriv(h1) * d_w_4;
            d_w[0]           = xs[0] * base;
            d_w[1]           = xs[1] * base;
            d_b[0]           = 1 * base;
        }
        // h2 derivatives
        {
            const float base = cog_sigmoid_deriv(h2) * d_w_5;
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
        float h1 = cog_sigmoid(n1);
        float n2 = xs[0] * w[2] + xs[1] * w[3] + b[1];
        float h2 = cog_sigmoid(n2);
        float o1 = h1 * w[4] + h2 * w[5] + b[2];
        float a1 = cog_sigmoid(o1);

        const float prediction = a1;
        sprintf(out + strlen(out), "prediction: %f, truth: %f, mse: %f\n", prediction, y_true[0],
                cog_mse(prediction, y_true[0]));
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
        printf("\033[32m[+] %s passed\033[0m\n", __FILE__);
    }

    return 0;
}
