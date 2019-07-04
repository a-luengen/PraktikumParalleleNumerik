#!/bin/bash

NAME=main22
CODE=$NAME.cu
SIZE=100

echo $CODE
echo "Compiling"
nvcc $CODE -o $NAME
