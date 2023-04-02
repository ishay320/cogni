#!/usr/bin/env bash
set -e

SOURCES=(./compare_to_math.c ./busses.c)
BUILD=./build/
INCLUDE=-I../
LINK=-lm
CFLAGS="-Wall -Wextra -Wshadow -pedantic -g -fno-omit-frame-pointer -fsanitize=address"

cd tests || exit 1

mkdir -p $BUILD
for FILE in "${SOURCES[@]}"; do
    TARGET_NAME="$(basename "${FILE}" .c).a"
    TARGETS+=("${TARGET_NAME}")
    gcc ${CFLAGS} ${FILE} ${INCLUDE} -o "${BUILD}${TARGET_NAME}" ${LINK}
done

for FILE in "${TARGETS[@]}"; do
    ./"${BUILD}${FILE}"
done
