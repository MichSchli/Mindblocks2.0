#!/bin/bash

GLOVE_DIR="test_data/embeddings"

echo "Downloading glove embeddings..."
[ -d $GLOVE_DIR ] || mkdir $GLOVE_DIR
wget -N -O $GLOVE_DIR/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

wget -N -O $GLOVE_DIR/glove.6B.100d.zip http://nlp.stanford.edu/data/glove.6B.100d.zip
unzip $GLOVE_DIR/glove.6B.100d.zip -d $GLOVE_DIR