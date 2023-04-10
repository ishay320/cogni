#ifndef DATABASE_H
#define DATABASE_H
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int error;

/* uses malloc on data variable */
error read_csv_f(const char* filename, float** data, size_t* columns, size_t* rows,
                 bool throw_first_row);
error get_csv_dimensions(FILE* csv_file, size_t* columns, size_t* rows);
#endif // DATABASE_H

#ifdef DATABASE_IMPLEMENTATION

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
    *data = (float*)malloc((sizeof **data) * (*columns) * (*rows));
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

#endif // DATABASE_IMPLEMENTATION
// TODO: add option to skip columns and read only columns
