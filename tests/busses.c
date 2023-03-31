#define COGNI_IMPLEMENTATION
#include "cogni.h"

#include <stdbool.h>
#include <string.h>

#define MAX_LINE_SIZE 1024
const char* g_filename = "data/busses.csv";

error get_csv_dimensions(FILE* csv_file, size_t* columns, size_t* rows)
{
    char c;
    c = (char)fgetc(csv_file);
    if (c == EOF) /* file empty */
    {
        fclose(csv_file);
        *columns = 0;
        *rows    = 0;
        return 0;
    }

    size_t comma_num = 0;
    while (c != EOF && c != '\n')
    {
        if (c == ',')
        {
            comma_num++;
        }
        c = (char)fgetc(csv_file);
    }

    size_t line_num    = 1; /* start from 1 because already read a line */
    size_t buffer_size = 0;
    char* buffer       = NULL;
    while (getline(&buffer, &buffer_size, csv_file) != -1)
    {
        ++line_num;
    }
    free(buffer);
    buffer = NULL;

    *columns = comma_num + 1;
    *rows    = line_num;

    return 0;
}

/* uses malloc on data */
error read_csv_f(const char* filename, float** data, size_t* columns, size_t* rows,
                 bool throw_first_row)
{
    FILE* csv_file = fopen(filename, "r");
    if (csv_file == NULL)
    {
        *columns = 0;
        *rows    = 0;
        fprintf(stderr, "ERROR: could not open csv file '%s': %s\n", filename, strerror(errno));
        return 1;
    }

    get_csv_dimensions(csv_file, columns, rows);
    if (rows == 0 && throw_first_row)
    {
        fclose(csv_file);
        printf("file '%s' empty\n", filename);
        return 1;
    }

    (*rows) -= throw_first_row;
    *data = malloc((sizeof **data) * (*columns) * (*rows));
    if (*data == NULL)
    {
        fclose(csv_file);
        fprintf(stderr, "ERROR: could not malloc data: %s", strerror(errno));
        return 1;
    }

    // reset file position
    rewind(csv_file);
    if (throw_first_row)
    {
        fscanf(csv_file, "%*[^\n]\n");
    }

    char* buffer = NULL;
    for (size_t line = 0; line < *rows; line++)
    {
        size_t buffer_size = 0;
        if (getline(&buffer, &buffer_size, csv_file) == -1)
        {
            fclose(csv_file);
            printf("ERROR: could not read line in file: %s\n", filename);
            return 1;
        }
        char* buffer_ptr = buffer;
        for (size_t i = 0; i < *columns; i++)
        {
            char* end_num                  = NULL;
            (*data)[line * (*columns) + i] = strtof(buffer_ptr, &end_num);
            buffer_ptr                     = end_num + 1;
        }
        free(buffer);
        buffer = NULL;
    }
    fclose(csv_file);

    return 0;
}

int main(int argc, char const* argv[])
{
    UNUSED(argc);
    UNUSED(argv);

    size_t columns, rows;
    float* xs;
    float* ys;
    const size_t y_stride = 4;

    // TODO: add option to skip columns and read only columns
    if (read_csv_f(g_filename, &xs, &columns, &rows, true) != 0)
    {
        return 1;
    }
    ys = &xs[3];

    Layer* l1 = cog_layer_init(2, 5);
    Layer* l2 = cog_layer_init(5, 3);
    Layer* l3 = cog_layer_init(3, 1);

    cog_layer_run(l1, xs);
    // TODO: get the out of the neuron to the layer
    cog_layer_run(l2, l1->neurons[0].out);
    cog_layer_run(l3, l2->neurons[0].out);

    float prediction = *l3->neurons[0].out;

#if 0
    printf("prediction: %f, truth: %f, mse: %f\n", prediction, ys[0 * y_stride],
           cog_mse(ys[0 * y_stride], prediction));
#endif

    const size_t epochs = 100;
    const float lr      = 0.000005;
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
#if 0
        cog_print_layer(l1, false, "l1");
        cog_print_layer(l2, false, "l2");
        cog_print_layer(l3, false, "l3");
#endif

        float d_mse = cog_mse_deriv(ys[0 * y_stride], prediction);
        cog_layer_backpropagate(l3, &d_mse);
        cog_layer_part_derive(l3);
        cog_layer_backpropagate(l2, l3->part_derive);
        cog_layer_part_derive(l2);
        cog_layer_backpropagate(l1, l2->part_derive);

        // Apply the derives
        cog_layer_apply_derives(l1, lr);
        cog_layer_apply_derives(l2, lr);
        cog_layer_apply_derives(l3, lr);

        // forward pass (prediction)
        cog_layer_run(l1, xs);
        cog_layer_run(l2, l1->neurons[0].out);
        cog_layer_run(l3, l2->neurons[0].out);

        prediction = *l3->neurons[0].out;

#if 0
        printf("prediction: %f, truth: %f, mse: %f\n", prediction, ys[0 * y_stride],
               cog_mse(ys[0 * y_stride], prediction));
#endif
    }

    const float mse   = cog_mse(ys[0], prediction);
    const float trues = ys[0];

    free(xs);
    cog_layer_destroy(l1);
    cog_layer_destroy(l2);
    cog_layer_destroy(l3);

    if (mse > 1)
    {
        printf(
            "\033[31m[-] %s test failed: the mse is bigger then 1 : %f prediction: %f true: %f\n",
            __FILE__, mse, prediction, trues);
    }
    else
    {
        printf("\033[32m[+] %s passed\033[0m\n", __FILE__);
    }
    return 0;
}
