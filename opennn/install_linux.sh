#!/bin/sh

cd
wget -O opennn.tar.gz https://github.com/Artelnics/opennn/archive/v4.0.tar.gz
tar -xvf opennn.tar.gz
cd opennn-4.0
cmake .
cmake -DCMAKE_TYPE_BUILD=Release .
make
