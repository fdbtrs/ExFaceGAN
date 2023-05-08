#!/usr/bin/env bash
DATASET="ExFaceGAN_SG3"

geteerinf -p "data_plots/"$DATASET -i "impostors.txt" -g "genuines.txt" -sp "data_plots/"$DATASET -e $DATASET
