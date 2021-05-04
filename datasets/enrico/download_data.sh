#!/bin/bash
# make dataset folder
mkdir -p dataset
# download data
wget https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv -P dataset
wget http://userinterfaces.aalto.fi/enrico/resources/screenshots.zip -P dataset
wget http://userinterfaces.aalto.fi/enrico/resources/wireframes.zip -P dataset
wget http://userinterfaces.aalto.fi/enrico/resources/hierarchies.zip -P dataset
wget http://userinterfaces.aalto.fi/enrico/resources/metadata.zip -P dataset
# unzip data
cd dataset
unzip screenshots.zip
unzip wireframes.zip
unzip hierarchies.zip
unzip metadata.zip
# remove archive files
rm screenshots.zip
rm wireframes.zip
rm hierarchies.zip
rm metadata.zip
cd ..