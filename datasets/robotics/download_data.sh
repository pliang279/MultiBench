#!/bin/bash

cd datasets/robotics
wget http://downloads.cs.stanford.edu/juno/triangle_real_data.zip -O _tmp.zip

unzip _tmp.zip
rm _tmp.zip
