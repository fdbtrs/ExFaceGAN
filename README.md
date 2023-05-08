# ExFaceGAN
# Exploring Identity Directions in GAN’s Learned Latent Space for Synthetic Identity Generation

![DIRGAN Overview](images/Overview_DIRGAN_Framework.png?raw=true)

## Requirements for DIRGAN
It is recommented creating a virtual environment with Python 3.6 and *`requirements.txt`*

## Pipeline for Disentangling Identity Information and Generate Images
1. Generate unconditional images
2. Create SVM training data
3. Train SVMs
4. Generate facial images
5. Train FR model

### 1. Generate Unconditional Images
Call:
```
CUDA_VISIBLE_DEVICES=0 python generate_SG_imgs.py --num_imgs 11000 --modelname "stylegan3_ffhq" --save_path "path/to/save/images_and_latents"
```
Align images via:
```
CUDA_VISIBLE_DEVICES=0 python MTCNN_alignment.py --in_folder "path/to/images_and_latents/images" --out_folder "path/to/images_and_latents/images_aligned"
```

### 2. Create SVM Training Data
Call:
```
python create_boundary_data.py --datadir "path/to/previous/images_and_latents" --save_path "path/to/save/boundary/data" --fr_path "path/tp/pretrained/FR/model" 
```

### 3. Train SVMs
Call:
```
python train_boundaries.py --datadir "path/to/previous/boundary/data" --save_path "path/to/save/boundaries"
```

### 4. Generate Facial Images
Call:
```
CUDA_VISIBLE_DEVICES=0 python create_dataset.py --latent_dir "path/to/previous/images_and_latents/w_latents" --boundary_dir "path/to/previous/boundaries" --modelname "stylegan3_ffhq" --output_path "path/to/save/new/dataset"
```
or set these parameters as default and call *`create_dataset.sh`* (change num_classes and offset) to generate images on 4 GPUs in parallel.

### 5. Train FR Model
Change *`config_baseline.py`* and *`run.sh`* to your preferences and execute:
```
run.sh
```


## Additional Scripts
*`analyse_dataset.py`* plots genuine-impostor-distributions and saves genuine and impostor scores for given datasets.

Given genuine, impostor scores, *`dataset_EER.sh`* calculates multiple metrics including EER, FMR, ...

*`impact_of_max_off.py`* allows editing one latent code and plots cosine similarity between edited images and the reference.