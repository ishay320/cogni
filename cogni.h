#ifndef COGNI_INCLUDE_H
#define COGNI_INCLUDE_H

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COGNI_DEF
#ifdef COGNI_STATIC
#define COGNI_DEF static
#else
#define COGNI_DEF extern
#endif
#endif

typedef int error;
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

COGNI_DEF float cog_mse(float x, float y);
COGNI_DEF float cog_mse_deriv(float truth, float pred);
COGNI_DEF float cog_sigmoid(float x);
COGNI_DEF float cog_sigmoid_deriv(float x);
COGNI_DEF void cog_print_arr(float* arr, size_t len, char* name);
COGNI_DEF error cog_write_weights(const char* path, const float* weights, size_t w_len,
                                  const float* bias, size_t b_len);
COGNI_DEF error cog_read_weights(const char* path, float* weights, size_t w_len, float* bias,
                                 size_t b_len);
COGNI_DEF Neuron* cog_neuron_init(float x[], float w[], float* b, float dw[], float* db, size_t len,
                                  activision fun, activision fun_derive, float* out);
COGNI_DEF void cog_neuron_destroy(Neuron* neuron);
COGNI_DEF float cog_calculate_linear(const float* w, const float* x, size_t len, float b);
COGNI_DEF float cog_neuron_forward(Neuron* neuron);
COGNI_DEF void cog_neuron_backpropagate(Neuron* neuron, float part_derive);
COGNI_DEF void cog_neuron_part_derive(Neuron* neuron, float* part_derives);
COGNI_DEF void cog_apply_derives(float* w, float* dw, size_t w_len, float* b, float* db,
                                 size_t b_len, float lr);

#endif // COGNI_INCLUDE_H

#ifdef COGNI_IMPLEMENTATION

#define COGNI_POW2(x) ((x) * (x))
#define UNUSED(var) (void)var

COGNI_DEF float cog_mse(float x, float y)
{
    return COGNI_POW2(x - y);
}

COGNI_DEF float cog_mse_deriv(float truth, float pred)
{
    return -2 * (truth - pred);
}

COGNI_DEF float cog_sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

COGNI_DEF float cog_sigmoid_deriv(float x)
{
    float sig = cog_sigmoid(x);
    return sig * (1.f - sig);
}

COGNI_DEF void cog_print_arr(float* arr, size_t len, char* name)
{
    printf("%s: ", name);
    for (size_t j = 0; j < len; j++)
    {
        printf("%f,", arr[j]);
    }
    printf("\n");
}

COGNI_DEF error cog_write_weights(const char* path, const float* weights, size_t w_len,
                                  const float* bias, size_t b_len)
{
    FILE* fp = fopen(path, "w");
    if (fp == 0)
    {
        fprintf(stderr, "could not open file '%s': %s\n", path, strerror(errno));
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

COGNI_DEF error cog_read_weights(const char* path, float* weights, size_t w_len, float* bias,
                                 size_t b_len)
{
    FILE* fp = fopen(path, "r");
    if (fp == 0)
    {
        fprintf(stderr, "could not open file '%s': %s\n", path, strerror(errno));
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

COGNI_DEF Neuron* cog_neuron_init(float x[], float w[], float* b, float dw[], float* db, size_t len,
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

COGNI_DEF void cog_neuron_destroy(Neuron* neuron)
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

COGNI_DEF float cog_calculate_linear(const float* w, const float* x, size_t len, float b)
{
    float sum = 0;
    for (size_t i = 0; i < len; i++)
    {
        sum += w[i] * x[i];
    }
    sum += b;
    return sum;
}

COGNI_DEF float cog_neuron_forward(Neuron* neuron)
{
    const float linear_out =
        cog_calculate_linear(neuron->w, neuron->x, neuron->w_len, *(neuron->b));
    const float activated = neuron->fun(linear_out);
    *neuron->out          = activated;
    return *neuron->out;
}

COGNI_DEF void cog_neuron_backpropagate(Neuron* neuron, float part_derive)
{
    neuron->base_derive = neuron->fun_derive(*neuron->out) * part_derive;
    for (size_t i = 0; i < neuron->w_len; i++)
    {
        neuron->dw[i] = neuron->base_derive * neuron->x[i];
    }
    *neuron->db = neuron->base_derive;
}

COGNI_DEF void cog_neuron_part_derive(Neuron* neuron, float* part_derives)
{
    for (size_t i = 0; i < neuron->w_len; i++)
    {
        part_derives[i] = neuron->base_derive * neuron->w[i];
    }
}

COGNI_DEF void cog_apply_derives(float* w, float* dw, size_t w_len, float* b, float* db,
                                 size_t b_len, float lr)
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

#endif // COGNI_IMPLEMENTATION
