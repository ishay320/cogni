#!/usr/bin/env bash

SOURCES=./test/compare_to_math.c
TARGET=compare_to_math.a
BUILD=./test/
INCLUDE=-I./
LINK=-lm
CFLAGS="-Wall -Wextra -pedantic -g"

gcc $CFLAGS $SOURCES $INCLUDE -o $BUILD$TARGET $LINK

cd $BUILD
if [[ $? == 0 ]]; then
    ./$TARGET
fi
