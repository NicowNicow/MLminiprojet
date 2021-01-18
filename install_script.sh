#!/bin/sh

#OpenNN setup
wget -O opennn.tar.gz https://github.com/Artelnics/opennn/archive/v4.0.tar.gz
tar -xvf opennn.tar.gz
cd opennn-4.0
find examples -name \*.cpp -exec sed -i "s/..\/data/data/g" {} \; # fixing data path
cmake .
cmake -DCMAKE_TYPE_BUILD=Release .
make
rm openn,.tar.gz

#Theano setup
cd ..
mkdir pip_packages
sudo apt install python3
sudo apt install python3-pip
sudo pip3 install Theano -t ~/pip_packages/
sudo pip3 install pydot -t ~/pip_packages/
export PYTHONPATH=~/pip_packages/
