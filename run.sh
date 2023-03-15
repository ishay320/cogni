#!/usr/bin/env bash

SOURCES=./src/cogni.c
TARGET=cogni.a
INCLUDE=-I./include

gcc $SOURCES $INCLUDE -o $TARGET

if [[ $? == 0 ]]; then
    ./$TARGET
fi
