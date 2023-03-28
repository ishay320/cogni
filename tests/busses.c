#define COGNI_IMPLEMENTATION
#include "cogni.h"

#include <stdbool.h>

#define MAX_LINE_SIZE 1024

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
    *data = malloc((sizeof **data) * (*columns) * (*rows));
    if (*data == NULL)
    {
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
    for (size_t line = 0 + throw_first_row; line < (*rows); line++)
    {
        size_t buffer_size = 0;
        if (getline(&buffer, &buffer_size, csv_file) == -1)
        {
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
    }
    free(buffer);

    return 0;
}

int main(int argc, char const* argv[])
{
    UNUSED(argc);
    UNUSED(argv);

    const char* filename = "busses.csv";
    size_t columns, rows;
    float* xs;

    if (read_csv_f(filename, &xs, &columns, &rows, false) != 0)
    {
        return 1;
    }

    free(xs);
    printf("\033[32m[+] %s passed\033[0m\n", __FILE__);
    return 0;
}
