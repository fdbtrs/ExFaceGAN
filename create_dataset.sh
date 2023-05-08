#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python create_dataset.py --num_classes 1250 --start_offset 0 &

CUDA_VISIBLE_DEVICES=1 python create_dataset.py --num_classes 1250 --start_offset 1250 &

CUDA_VISIBLE_DEVICES=2 python create_dataset.py --num_classes 1250 --start_offset 2500 &

CUDA_VISIBLE_DEVICES=3 python create_dataset.py --num_classes 1250 --start_offset 3750
