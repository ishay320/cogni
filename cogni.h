#ifndef COGNI_INCLUDE_H
#define COGNI_INCLUDE_H

#include <errno.h>
#include <math.h>
#include <stdarg.h>
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

typedef struct Layer
{
    Neuron* neurons;
    float* outputs;
    float* part_derive;
    size_t len;
    struct Layer* last_layer;
} Layer;

COGNI_DEF float cog_mse(float x, float y);
COGNI_DEF float cog_mse_deriv(float truth, float pred);
COGNI_DEF float cog_sigmoid(float x);
COGNI_DEF float cog_sigmoid_deriv(float x);
COGNI_DEF float cog_relu(float x);
COGNI_DEF float cog_relu_deriv(float x);
COGNI_DEF float cog_lrelu(float x);
COGNI_DEF float cog_lrelu_deriv(float x);

COGNI_DEF void cog_print_arr(float* arr, size_t len, const char* format, ...);
COGNI_DEF error cog_write_weights(const char* path, const float* weights, size_t w_len,
                                  const float* bias, size_t b_len);
COGNI_DEF error cog_read_weights(const char* path, float* weights, size_t w_len, float* bias,
                                 size_t b_len);
COGNI_DEF Neuron* cog_neuron_init_m(float x[], float w[], float* b, float dw[], float* db,
                                    size_t len, activision fun, activision fun_derive, float* out);
COGNI_DEF Neuron* cog_neuron_init(Neuron* neuron, float x[], float w[], float* b, float dw[],
                                  float* db, size_t len, activision fun, activision fun_derive,
                                  float* out);
COGNI_DEF void cog_neuron_destroy(Neuron* neuron);
COGNI_DEF float cog_calculate_linear(const float* w, const float* x, size_t len, float b);
COGNI_DEF float cog_neuron_forward(Neuron* neuron);
COGNI_DEF void cog_neuron_backpropagate(Neuron* neuron, float part_derive);
COGNI_DEF void cog_neuron_part_derive(Neuron* neuron, float* part_derives);
COGNI_DEF void cog_apply_derives(float* w, float* dw, size_t w_len, float* b, float* db,
                                 size_t b_len, float lr);

COGNI_DEF void cog_array_rand_f(float* array, size_t len, float min, float max);
COGNI_DEF void cog_layer_destroy(Layer* layer);
COGNI_DEF Layer* cog_layer_init(size_t in_features, size_t out_features);
COGNI_DEF void cog_layer_run(Layer* layer, const float* x);
COGNI_DEF void cog_layer_backpropagate(Layer* layer, float* partial_derive);
COGNI_DEF void cog_layer_part_derive(Layer* layer);
COGNI_DEF void cog_layer_apply_derives(Layer* layer, float lr);

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

COGNI_DEF float cog_relu(float x)
{
    return x * (x > 0);
}

COGNI_DEF float cog_relu_deriv(float x)
{
    return x > 0;
}

COGNI_DEF float cog_lrelu(float x)
{
    return (x > 0) ? x : x * 0.01;
}

COGNI_DEF float cog_lrelu_deriv(float x)
{
    return (x > 0) ? 1 : 0.01;
}

COGNI_DEF void cog_print_arr(float* arr, size_t len, const char* format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stderr, format, argptr);
    va_end(argptr);

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

COGNI_DEF Neuron* cog_neuron_init_m(float x[], float w[], float* b, float dw[], float* db,
                                    size_t len, activision fun, activision fun_derive, float* out)
{
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (neuron == 0)
    {
        fprintf(stderr, "[ERROR] Could not allocate memory for neuron.\n");
        return NULL;
    }
    return cog_neuron_init(neuron, x, w, b, dw, db, len, fun, fun_derive, out);
}

COGNI_DEF Neuron* cog_neuron_init(Neuron* neuron, float x[], float w[], float* b, float dw[],
                                  float* db, size_t len, activision fun, activision fun_derive,
                                  float* out)
{
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

COGNI_DEF void cog_array_rand_f(float* array, size_t len, float min, float max)
{
    max -= min;
    for (size_t i = 0; i < len; i++)
    {
        array[i] = (((float)rand() / (float)(RAND_MAX)) * max) + min;
    }
}

COGNI_DEF void cog_layer_destroy(Layer* layer)
{
    if (layer == NULL)
    {
        printf("ERROR: cannot double free\n");
        return;
    }

    if (layer->len == 0)
    {
        free(layer);
        return;
    }

    free(layer->neurons[0].w);
    free(layer->neurons[0].x);
    free(layer->neurons[0].b);
    free(layer->neurons[0].dw);
    free(layer->neurons[0].db);
    free(layer->neurons[0].out);

    free(layer->neurons);
    free(layer->part_derive);
    free(layer);
}

/* use malloc on return */
COGNI_DEF Layer* cog_layer_init(size_t in_features, size_t out_features)
{
    Layer* layer = malloc(sizeof(Layer));
    if (layer == NULL)
    {
        fprintf(stderr, "ERROR: could not malloc layer\n");
        return NULL;
    }

    layer->len         = out_features;
    layer->neurons     = malloc(sizeof(Neuron) * out_features);
    layer->part_derive = malloc(sizeof(float) * in_features * out_features);
    float* w           = malloc(sizeof(float) * in_features * out_features);
    float* x           = malloc(sizeof(float) * in_features);
    float* b           = malloc(sizeof(float) * out_features);
    float* dw          = malloc(sizeof(float) * in_features * out_features);
    float* db          = malloc(sizeof(float) * out_features);
    float* out         = malloc(sizeof(float) * out_features);
    if (layer == NULL || layer->part_derive == NULL || w == NULL || x == NULL || b == NULL ||
        dw == NULL || db == NULL || out == NULL)
    {
        fprintf(stderr, "ERROR: could not malloc neurons data\n");
        // TODO: check NULL for double free (define array and for loop)
        free(layer->neurons);
        free(layer->part_derive);
        free(layer);
        free(w);
        free(x);
        free(b);
        free(dw);
        free(db);
        free(out);
        return NULL;
    }

    cog_array_rand_f(w, in_features * out_features, 0, 1);
    cog_array_rand_f(b, out_features, 0, 1);

    for (size_t i = 0; i < out_features; i++)
    {
        cog_neuron_init(&layer->neurons[i], x, &w[i * in_features], &b[i], dw, db, in_features,
                        &cog_lrelu, &cog_lrelu_deriv, &out[i]);
    }

    return layer;
}

COGNI_DEF void cog_layer_run(Layer* layer, const float* x)
{
    if (layer->len == 0)
    {
        return;
    }

    // TODO: move the x of the neuron to the layer and pass to the forward the x
    memcpy(layer->neurons[0].x, x, (sizeof *x) * (layer->neurons[0].w_len));
    for (size_t i = 0; i < layer->len; i++)
    {
        cog_neuron_forward(&layer->neurons[i]);
    }
}

COGNI_DEF void cog_layer_backpropagate(Layer* layer, float* partial_derive)
{
    for (size_t i = 0; i < layer->len; i++)
    {
        cog_neuron_backpropagate(&layer->neurons[i], partial_derive[i * layer->neurons[i].w_len]);
    }
}

COGNI_DEF void cog_layer_part_derive(Layer* layer)
{
    for (size_t i = 0; i < layer->len; i++)
    {
        cog_neuron_part_derive(&layer->neurons[i],
                               &layer->part_derive[i * layer->neurons[i].w_len]);
    }
}

COGNI_DEF void cog_layer_apply_derives(Layer* layer, float lr)
{
    for (size_t i = 0; i < layer->len; i++)
    {
        // can be done with the first and all the size
        cog_apply_derives(layer->neurons[i].w, layer->neurons[i].dw, layer->neurons[i].w_len,
                          layer->neurons[i].b, layer->neurons[i].db, 1, lr);
    }
}

#endif // COGNI_IMPLEMENTATION
