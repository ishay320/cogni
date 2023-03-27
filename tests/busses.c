#define COGNI_IMPLEMENTATION
#include "cogni.h"

#define MAX_LINE_SIZE 1024

error get_csv_dimensions(const char* filename, size_t* columns, size_t* rows)
{
    FILE* csv_file = fopen(filename, "r");
    if (csv_file == NULL)
    {
        *columns = 0;
        *rows    = 0;
        fprintf(stderr, "ERROR: could not open csv file '%s': %s\n", filename, strerror(errno));
        return 1;
    }

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

    fclose(csv_file);
    return 0;
}

int main(int argc, char const* argv[])
{
    UNUSED(argc);
    UNUSED(argv);

    const char* filename = "busses.csv";
    size_t x, y;
    get_csv_dimensions(filename, &x, &y);

    printf("\033[32m[+] %s passed\033[0m\n", __FILE__);
    return 0;
}
