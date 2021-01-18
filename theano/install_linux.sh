#!/bin/sh

cd
mkdir pip_packages
sudo apt install python3
sudo apt install python3-pip
sudo pip3 install Theano -t ~/pip_packages/
sudo pip3 install pydot -t ~/pip_packages/
export PYTHONPATH=~/pip_packages/
