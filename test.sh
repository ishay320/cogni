#!/usr/bin/env bash

SOURCES=./compare_to_math.c
TARGET=compare_to_math.a
BUILD=./build/
INCLUDE=-I../
LINK=-lm
CFLAGS="-Wall -Wextra -pedantic -g"

cd tests || exit 1

mkdir -p $BUILD
gcc $CFLAGS $SOURCES $INCLUDE -o $BUILD$TARGET $LINK

if [[ $? == 0 ]]; then
    ./$BUILD$TARGET
fi
