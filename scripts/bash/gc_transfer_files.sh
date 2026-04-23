#!/bin/bash

USER=pberger
GROUP=cellmembrane

rsync -avh --partial --progress \
  colab_lipid_gnn_subset.zip \
  $USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/lipid_data/

#rsync -avh --partial --progress \
#  results/properties/ \
#  $USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/lipid_data/results/properties/