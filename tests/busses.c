#define COGNI_IMPLEMENTATION
#include "cogni.h"

#define DATABASE_IMPLEMENTATION
#include "database.h"

const char* g_filename = "data/busses.csv";

int main(int argc, char const* argv[])
{
    UNUSED(argc);
    UNUSED(argv);

    size_t columns, rows;
    float* xs;
    float* ys;
    const size_t y_stride = 5;

    if (read_csv_f(g_filename, &xs, &columns, &rows, true) != 0)
    {
        return 1;
    }
    ys = &xs[y_stride - 1];

    LayerFC* l1 = cog_layer_init(4, 8);
    LayerFC* l2 = cog_layer_init(8, 7);
    LayerFC* l3 = cog_layer_init(7, 5);
    LayerFC* l4 = cog_layer_init(5, 1);

    LayerActivision lrelu = cog_layer_activision_init(L_RELU);

#if defined(READ_WEIGHTS) || defined(WRITE_WEIGHTS)
    FILE* fp = fopen("busses.w", "r+");
#endif
#if defined(READ_WEIGHTS)
    cog_read_weights_p(fp, l1->neurons[0].w, l1->len * l1->neurons[0].w_len, l1->neurons[0].b,
                       l1->len);
    cog_read_weights_p(fp, l2->neurons[0].w, l2->len * l2->neurons[0].w_len, l2->neurons[0].b,
                       l2->len);
    cog_read_weights_p(fp, l3->neurons[0].w, l3->len * l3->neurons[0].w_len, l3->neurons[0].b,
                       l3->len);
    cog_read_weights_p(fp, l4->neurons[0].w, l4->len * l4->neurons[0].w_len, l4->neurons[0].b,
                       l4->len);
    rewind(fp);
#endif
    FILE* out = fopen("rand.log", "w+");

    float prediction    = 0;
    const size_t epochs = 500;
    const float lr      = 0.00005;
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float avg_mse = 0;
        for (size_t pos = 0; pos < rows; pos++)
        {
            // forward pass (prediction)
            cog_layer_run(l1, xs + (pos * y_stride));
            cog_layer_activate(lrelu, l1);
            cog_layer_run(l2, l1->outputs);
            cog_layer_activate(lrelu, l2);
            cog_layer_run(l3, l2->outputs);
            cog_layer_activate(lrelu, l3);
            cog_layer_run(l4, l3->outputs);
            // cog_layer_activate(lrelu, l4);

            prediction        = l4->outputs[0];
            const float trues = ys[pos * y_stride];
            const float mse   = cog_mse(trues, prediction);
            avg_mse += mse / rows;

#if 0
        // print error
        if ((epoch % (10000 * rows)) == 0)
        {
            printf("prediction: %f, truth: %f, mse: %f\n", prediction,
                   ys[(epoch % rows) * y_stride],
                   cog_mse(ys[(epoch % rows) * y_stride], prediction));
        }
#endif
#if 0 
        // print layers data
        cog_print_layer(l1, false, "l1");
        cog_print_layer(l2, false, "l2");
        cog_print_layer(l3, false, "l3");
        cog_print_layer(l4, false, "l4");
#endif

            cog_layer_zero_grad(l1);
            cog_layer_zero_grad(l2);
            cog_layer_zero_grad(l3);
            cog_layer_zero_grad(l4);

            // Derive layers
            const float d_mse = cog_mse_deriv(trues, prediction);
            cog_layer_backpropagate(l4, &d_mse);
            cog_layer_part_derive(l4);
            cog_layer_backpropagate(l3, l4->part_derive);
            cog_layer_part_derive(l3);
            cog_layer_backpropagate(l2, l3->part_derive);
            cog_layer_part_derive(l2);
            cog_layer_backpropagate(l1, l2->part_derive);

            // Apply the derives
            cog_layer_apply_derives(l1, lr);
            cog_layer_apply_derives(l2, lr);
            cog_layer_apply_derives(l3, lr);
            cog_layer_apply_derives(l4, lr);
        }
        fprintf(out, "%f\n", avg_mse);
    }
    fclose(out);

#if defined(WRITE_WEIGHTS)
    cog_write_weights_p(fp, l1->neurons[0].w, l1->len * l1->neurons[0].w_len, l1->neurons[0].b,
                        l1->len);
    cog_write_weights_p(fp, l2->neurons[0].w, l2->len * l2->neurons[0].w_len, l2->neurons[0].b,
                        l2->len);
    cog_write_weights_p(fp, l3->neurons[0].w, l3->len * l3->neurons[0].w_len, l3->neurons[0].b,
                        l3->len);
    cog_write_weights_p(fp, l4->neurons[0].w, l4->len * l4->neurons[0].w_len, l4->neurons[0].b,
                        l4->len);
#endif
#if defined(READ_WEIGHTS) || defined(WRITE_WEIGHTS)
    fclose(fp);
#endif

    float avg_mse = 0;
    for (size_t i = 0; i < rows; i++)
    {
        cog_layer_run(l1, xs + (i * y_stride));
        cog_layer_activate(lrelu, l1);
        cog_layer_run(l2, l1->outputs);
        cog_layer_activate(lrelu, l2);
        cog_layer_run(l3, l2->outputs);
        cog_layer_activate(lrelu, l3);
        cog_layer_run(l4, l3->outputs);
        // cog_layer_activate(lrelu, l4);

        prediction = *l4->outputs;

        const float trues = ys[i * y_stride];
        const float mse   = cog_mse(trues, prediction);
        avg_mse += mse / rows;
#if 0
        printf("prediction: %f, truth: %f, mse: %f, params: %f %f %f %f\n", prediction, trues, mse,
               xs[(i * y_stride)], xs[(i * y_stride) + 1], xs[(i * y_stride) + 2],
               xs[(i * y_stride) + 3]);
#endif
    }

    free(xs);
    cog_layer_destroy(l1);
    cog_layer_destroy(l2);
    cog_layer_destroy(l3);
    cog_layer_destroy(l4);

    const int max_mse = 190;
    if (isnan(prediction))
    {
        printf("\033[31m[-] %s test failed: prediction failed: %f\n", __FILE__, prediction);
    }
    else if (avg_mse > max_mse)
    {
        printf("\033[31m[-] %s test failed: the avg mse is bigger then %d : %f , last time it was "
               "182.870529\n",
               __FILE__, max_mse, avg_mse);
    }
    else
    {
        printf("\033[32m[+] %s \t\tpassed %f\033[0m\n", __FILE__, avg_mse);
    }
    return 0;
}
