#!/bin/bash
# ----------------------------------------------------------------
# Script for generating plots
# Usage: $0 <log file> <out file>
# ----------------------------------------------------------------
INPUT_FILE_PATH=""
OUTPUT_FILE_PATH="out.png"

USE_FIRST_AS_X=false
START_COLUMN=1
X_AXIS=0

usage() {
    echo "Usage: $0 <log file> <out file>"
    echo "  -f --first  use the first column as the x axis, default to line number"
    echo "  -h --help   print this help message"
    echo "  -i --input  input file path"
    echo "  -o --out    output file path, default to out.png"
    echo "  -s --skip   skip on n column"
}

while (($# > 0)); do
    case "${1}" in
    -f | --first)
        USE_FIRST_AS_X=true
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    -i | --input)
        numOfArgs=1
        if (($# < numOfArgs + 1)); then
            echo "argument $1 is empty"
            shift $#
            exit 1
        else
            INPUT_FILE_PATH=$2
            shift $((numOfArgs + 1))
        fi
        ;;
    -o | --out)
        numOfArgs=1
        if (($# < numOfArgs + 1)); then
            echo "argument $1 is empty"
            shift $#
            exit 1
        else
            OUTPUT_FILE_PATH=$2
            shift $((numOfArgs + 1))
        fi
        ;;
    -s | --skip)
        numOfArgs=1
        if (($# < numOfArgs + 1)); then
            echo "argument $1 is empty"
            shift $#
            exit 1
        else
            SKIP_NUM=$2
            START_COLUMN=$((SKIP_NUM + 1))
            shift $((numOfArgs + 1))
        fi
        ;;
    *) # unknown flag/switch
        usage
        exit 1
        ;;
    esac
done

if [[ -z "${INPUT_FILE_PATH}" ]]; then
    usage
    exit 1
fi

# Determine number of columns in data file
NUM_COLS=$(awk '{print NF}' "$INPUT_FILE_PATH" | sort -nu | tail -n 1)

# Set up Gnuplot commands
GNUPLOT_COMMANDS=$(
    cat <<-END
set term png
set output '${OUTPUT_FILE_PATH}'
set key outside
set xlabel 'X Axis'
set ylabel 'Y Axis'
plot 
END
)

if [[ ${USE_FIRST_AS_X} == true ]]; then
    if (($(echo "${START_COLUMN} == 1" | bc -l))); then
        ((START_COLUMN = 2))
    fi
    X_AXIS=1
fi

# Loop over columns and add plot command for each one
for ((i = ${START_COLUMN}; i <= "${NUM_COLS}"; i++)); do
    GNUPLOT_COMMANDS+=" '$INPUT_FILE_PATH' using ${X_AXIS}:$i with lines title 'Column $i', "
done

# Remove trailing comma from last plot command
GNUPLOT_COMMANDS="${GNUPLOT_COMMANDS%, }"

# Execute Gnuplot commands
echo "$GNUPLOT_COMMANDS" >t
echo "$GNUPLOT_COMMANDS" | gnuplot

# TODO: add column name to csv title if exists
# TODO: add  <-s --size> for png size
# TODO: add  option to show on the screen
# TODO: add option to choose the x value
