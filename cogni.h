#ifndef COGNI_INCLUDE_H
#define COGNI_INCLUDE_H

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
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
    // weights and bias
    float* w;
    size_t w_len;
    float* b;

    // derivatives
    float base_derive;
    float* dw;
    float* db;
} Neuron;

typedef struct LayerFC
{
    Neuron* neurons;
    float* inputs;
    float* outputs;
    float* part_derive;
    size_t len;

    activision last_activision;
} LayerFC;

typedef struct LayerActivision
{
    activision fun;
    activision fun_derive;
} LayerActivision;

typedef enum
{
    NONE = 0,
    RELU,
    L_RELU,
    SIGMOID,
    ACTIVISION_LEN
} Activision_type;

/* Functions */
COGNI_DEF float cog_mse(float x, float y);
COGNI_DEF float cog_mse_deriv(float truth, float pred);
COGNI_DEF float cog_sigmoid(float x);
COGNI_DEF float cog_sigmoid_deriv(float x);
COGNI_DEF float cog_relu(float x);
COGNI_DEF float cog_relu_deriv(float x);
COGNI_DEF float cog_lrelu(float x);
COGNI_DEF float cog_lrelu_deriv(float x);

/* Files io */
COGNI_DEF error cog_write_weights(const char* path, const float* weights, size_t w_len,
                                  const float* bias, size_t b_len);
COGNI_DEF error cog_write_weights_p(FILE* fp, const float* weights, size_t w_len, const float* bias,
                                    size_t b_len);
COGNI_DEF error cog_read_weights(const char* path, float* weights, size_t w_len, float* bias,
                                 size_t b_len);
COGNI_DEF error cog_read_weights_p(FILE* fp, float* weights, size_t w_len, float* bias,
                                   size_t b_len);

/* Neurons */
// Neuron init using malloc - use neuron_destroy
COGNI_DEF Neuron* cog_neuron_init_m(float w[], float* b, float dw[], float* db, size_t w_len);
COGNI_DEF Neuron* cog_neuron_init(Neuron* neuron, float w[], float* b, float dw[], float* db,
                                  size_t w_len);
COGNI_DEF void cog_neuron_destroy(Neuron* neuron);
COGNI_DEF float cog_calculate_linear(const float* w, const float* x, size_t len, float b);
COGNI_DEF float cog_neuron_forward(Neuron* neuron, const float* xs);

/* Derivatives */
COGNI_DEF void cog_fun_backpropagate(Neuron* neuron, activision fun, float last_out,
                                     float part_derive);
COGNI_DEF void cog_neuron_backpropagate(Neuron* neuron, const float* xs);

COGNI_DEF void cog_neuron_part_derive(Neuron* neuron, float* part_derives);

COGNI_DEF void cog_apply_derives(float* w, float* dw, size_t w_len, float* b, float* db,
                                 size_t b_len, float lr);

/* Layers */
COGNI_DEF LayerFC* cog_layer_init(size_t in_features, size_t out_features);
COGNI_DEF void cog_layer_destroy(LayerFC* layer);
COGNI_DEF float* cog_layer_run(LayerFC* layer, const float* xs);
COGNI_DEF void cog_layer_zero_grad(LayerFC* layer);
COGNI_DEF void cog_layer_backpropagate(LayerFC* layer, const float* partial_derive);
COGNI_DEF void cog_layer_part_derive(LayerFC* layer);
COGNI_DEF void cog_layer_apply_derives(LayerFC* layer, float lr);

COGNI_DEF LayerActivision cog_layer_activision_init(Activision_type type);
COGNI_DEF float* cog_layer_activate(const LayerActivision fun, LayerFC* layer);

/* Printing and debug */
COGNI_DEF void cog_print_layer(const LayerFC* layer, bool print_derive, const char* layer_name);
COGNI_DEF void cog_print_array(float* array, size_t len, const char* format, ...);

COGNI_DEF void cog_array_rand_f(float* array, size_t len, float min, float max);

#endif // COGNI_INCLUDE_H

#ifdef COGNI_IMPLEMENTATION

#define COGNI_POW2(x) ((x) * (x))
#define UNUSED(var) (void)var

static const struct
{
    activision fun;
    activision fun_derive;
} c_activision_index[] = {
    [NONE]    = {.fun = NULL, .fun_derive = NULL},
    [RELU]    = {.fun = cog_relu, .fun_derive = cog_relu_deriv},
    [L_RELU]  = {.fun = cog_lrelu, .fun_derive = cog_lrelu_deriv},
    [SIGMOID] = {.fun = cog_sigmoid, .fun_derive = cog_sigmoid_deriv},
};

_Static_assert((sizeof c_activision_index) / (sizeof *c_activision_index) == ACTIVISION_LEN,
               "ERROR: Please update the index of activision");

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

COGNI_DEF void cog_fun_backpropagate(Neuron* neuron, activision fun, float last_out,
                                     float part_derive)
{
    // TODO: remove this with function pointer
    if (fun == NULL)
    {
        neuron->base_derive = part_derive;
        return;
    }

    neuron->base_derive = fun(last_out) * part_derive;
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

    error err = cog_write_weights_p(fp, weights, w_len, bias, b_len);

    fclose(fp);
    return err;
}

COGNI_DEF error cog_write_weights_p(FILE* fp, const float* weights, size_t w_len, const float* bias,
                                    size_t b_len)
{
    for (size_t i = 0; i < w_len; i++)
    {
        fprintf(fp, "%a ", weights[i]);
    }
    fprintf(fp, "\n");
    for (size_t i = 0; i < b_len; i++)
    {
        fprintf(fp, "%a ", bias[i]);
    }
    fprintf(fp, "\n");
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
    error err = cog_read_weights_p(fp, weights, w_len, bias, b_len);

    fclose(fp);
    return err;
}

COGNI_DEF error cog_read_weights_p(FILE* fp, float* weights, size_t w_len, float* bias,
                                   size_t b_len)
{
    for (size_t i = 0; i < w_len; i++)
    {
        fscanf(fp, "%a ", &weights[i]);
    }
    fscanf(fp, "\n");
    for (size_t i = 0; i < b_len; i++)
    {
        fscanf(fp, "%a ", &bias[i]);
    }
    fscanf(fp, "\n");
    return 0;
}

COGNI_DEF Neuron* cog_neuron_init_m(float w[], float* b, float dw[], float* db, size_t w_len)
{
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (neuron == 0)
    {
        fprintf(stderr, "[ERROR] Could not allocate memory for neuron.\n");
        return NULL;
    }
    return cog_neuron_init(neuron, w, b, dw, db, w_len);
}

COGNI_DEF Neuron* cog_neuron_init(Neuron* neuron, float w[], float* b, float dw[], float* db,
                                  size_t w_len)
{
    neuron->w     = w;
    neuron->w_len = w_len;
    neuron->b     = b;
    neuron->dw    = dw;
    neuron->db    = db;

    neuron->base_derive = 0;

    return neuron;
}

COGNI_DEF void cog_neuron_destroy(Neuron* neuron)
{
    free(neuron);
}

COGNI_DEF float cog_calculate_linear(const float* w, const float* x, size_t len, float b)
{
    float sum = 0.f;
    for (size_t i = 0; i < len; i++)
    {
        sum += w[i] * x[i];
    }
    sum += b;
    return sum;
}

COGNI_DEF float cog_neuron_forward(Neuron* neuron, const float* xs)
{
    const float linear_out = cog_calculate_linear(neuron->w, xs, neuron->w_len, *(neuron->b));
    return linear_out;
}

COGNI_DEF void cog_neuron_backpropagate(Neuron* neuron, const float* xs)
{
    for (size_t i = 0; i < neuron->w_len; i++)
    {
        neuron->dw[i] = neuron->base_derive * xs[i];
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

COGNI_DEF void cog_layer_zero_grad(LayerFC* layer)
{
    memset(layer->neurons[0].dw, 0,
           (sizeof layer->neurons[0].dw[0]) * layer->neurons[0].w_len * layer->len);
    memset(layer->neurons[0].db, 0, (sizeof layer->neurons[0].db[0]) * layer->len);
}

/* use malloc on return value - use layer_destroy*/
COGNI_DEF LayerFC* cog_layer_init(size_t in_features, size_t out_features)
{
    LayerFC* layer = malloc(sizeof(LayerFC));
    if (layer == NULL)
    {
        fprintf(stderr, "ERROR: could not malloc layer\n");
        return NULL;
    }

    layer->last_activision = NULL;
    layer->len             = out_features;
    layer->neurons         = malloc(sizeof(Neuron) * out_features);
    layer->part_derive     = malloc(sizeof(float) * in_features * out_features);
    float* w               = malloc(sizeof(float) * in_features * out_features);
    layer->inputs          = malloc(sizeof(float) * in_features);
    float* b               = malloc(sizeof(float) * out_features);
    float* dw              = malloc(sizeof(float) * in_features * out_features);
    float* db              = malloc(sizeof(float) * out_features);
    layer->outputs         = malloc(sizeof(float) * out_features);
    if (layer == NULL || layer->part_derive == NULL || w == NULL || layer->inputs == NULL ||
        b == NULL || dw == NULL || db == NULL || layer->outputs == NULL)
    {
        fprintf(stderr, "ERROR: could not malloc neurons data\n");
        // TODO: check NULL for double free (define array and for loop)
        free(layer->neurons);
        free(layer->part_derive);
        free(layer->inputs);
        free(layer->outputs);
        free(layer);
        free(w);
        free(b);
        free(dw);
        free(db);
        return NULL;
    }

    cog_array_rand_f(w, in_features * out_features, 0, 1);
    cog_array_rand_f(b, out_features, 0, 1);

    for (size_t i = 0; i < out_features; i++)
    {
        cog_neuron_init(&layer->neurons[i], &w[i * in_features], &b[i], &dw[i * in_features],
                        &db[i], in_features);
    }

    return layer;
}

COGNI_DEF void cog_layer_destroy(LayerFC* layer)
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
    free(layer->neurons[0].b);
    free(layer->neurons[0].dw);
    free(layer->neurons[0].db);
    free(layer->outputs);

    free(layer->inputs);
    free(layer->neurons);
    free(layer->part_derive);
    free(layer);
}

COGNI_DEF float* cog_layer_run(LayerFC* layer, const float* xs)
{
    if (layer->len == 0)
    {
        return NULL;
    }

    layer->last_activision = NULL;
    // TODO: check option to save the ref to the xs - needed only for backprop
    memcpy(layer->inputs, xs, (sizeof *xs) * (layer->neurons[0].w_len));
    for (size_t i = 0; i < layer->len; i++)
    {
        layer->outputs[i] = cog_neuron_forward(&layer->neurons[i], xs);
    }

    return layer->outputs;
}

COGNI_DEF void cog_layer_backpropagate(LayerFC* layer, const float* partial_derive)
{
    for (size_t n = 0; n < layer->len; n++)
    {
        cog_fun_backpropagate(&layer->neurons[n], layer->last_activision, layer->outputs[n],
                              partial_derive[n]);
        cog_neuron_backpropagate(&layer->neurons[n], layer->inputs);
    }
}

COGNI_DEF void cog_layer_part_derive(LayerFC* layer)
{
    for (size_t i = 0; i < layer->len; i++)
    {
        cog_neuron_part_derive(&layer->neurons[i],
                               &layer->part_derive[i * layer->neurons[i].w_len]);
    }
}

COGNI_DEF void cog_layer_apply_derives(LayerFC* layer, float lr)
{
    for (size_t i = 0; i < layer->len; i++)
    {
        // can be done with the first and all the size
        cog_apply_derives(layer->neurons[i].w, layer->neurons[i].dw, layer->neurons[i].w_len,
                          layer->neurons[i].b, layer->neurons[i].db, 1, lr);
    }
}

COGNI_DEF LayerActivision cog_layer_activision_init(Activision_type type)
{
    LayerActivision layer = {.fun        = c_activision_index[type].fun,
                             .fun_derive = c_activision_index[type].fun_derive};
    return layer;
}

COGNI_DEF float* cog_layer_activate(const LayerActivision fun, LayerFC* layer)
{
    for (size_t i = 0; i < layer->len; i++)
    {
        layer->outputs[i] = fun.fun(layer->outputs[i]);
    }

    layer->last_activision = fun.fun_derive;
    return layer->outputs;
}

COGNI_DEF void cog_print_array(float* array, size_t len, const char* format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stderr, format, argptr);
    va_end(argptr);

    for (size_t j = 0; j < len; j++)
    {
        printf("%f,", array[j]);
    }
    printf("\n");
}

COGNI_DEF void cog_print_layer(const LayerFC* layer, bool print_derive, const char* layer_name)
{
    printf("%s:\n", layer_name);
    printf("| ");
    for (size_t i = 0; i < layer->neurons[0].w_len; i++)
    {
        printf("w%-13ld", i);
    }
    printf("b%-13s|\n", "");
    for (size_t n = 0; n < layer->len; n++)
    {
        printf("| ");
        for (size_t j = 0; j < layer->neurons[n].w_len; j++)
        {
            printf("%-13.6f ", layer->neurons[n].w[j]);
        }
        printf("%-13.6f |\n", *layer->neurons[n].b);
    }
    if (print_derive)
    {
        printf(" ");
        for (size_t i = 0; i < layer->neurons[0].w_len; i++)
        {
            printf("-----------------");
        }
        printf("\n");
        for (size_t n = 0; n < layer->len; n++)
        {
            printf("| ");
            for (size_t j = 0; j < layer->neurons[n].w_len; j++)
            {
                printf("%-13.6f ", layer->neurons[n].dw[j]);
            }
            printf("%-13.6f |\n", *layer->neurons[n].db);
        }
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
#endif // COGNI_IMPLEMENTATION
