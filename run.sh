#!/usr/bin/env bash

SOURCES=./src/cogni.c
TARGET=cogni.a
INCLUDE=-I./include
LINK=-lm

gcc $SOURCES $INCLUDE -o $TARGET $LINK

if [[ $? == 0 ]]; then
    ./$TARGET
fi
