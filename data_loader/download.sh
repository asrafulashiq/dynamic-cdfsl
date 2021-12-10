#!/bin/bash
# install gdown for downloading from google-drive link
pip install -U --no-cache-dir gdown --pre
# sudo apt install unzip

prev_directory=$(pwd) # current directory

mkdir -p data/cdfsl
cd data/cdfsl

# ISIC
gdown "https://drive.google.com/uc?id=1FhN8vgg6g0Vm6d-Q3IjeueRiOqU8SUkY"
unzip -qq ISIC.zip
rm -rf __MACOSX
rm ISIC.zip

#miniImageNet
gdown "https://drive.google.com/uc?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY"
mkdir mini-ImageNet
tar -zxf mini-imagenet.tar.gz -C mini-ImageNet/
rm mini-imagenet.tar.gz

# CropDisease
gdown "https://drive.google.com/uc?id=1UlJqQwG5e4PEHnQkBGiD8bqTUNW8t0c0"
unzip CropDiseases.zip
rm CropDiseases.zip
rm -rf __MACOSX

# EuroSAT
# gdown "https://drive.google.com/uc?id=1FYZvuBePf2tuEsEaBCsACtIHi6eFpSwe"
wget "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
unzip EuroSAT.zip
rm -rf __MACOSX
mkdir EuroSAT
mv -f 2750 EuroSAT/
rm EuroSAT.zip

# CUB200
gdown "https://drive.google.com/uc?id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx"
mkdir CUB_200_2011
tar -zxf images.tgz
mv -f images CUB_200_2011
rm images.tgz

# remove if there are other files
rm *.gz
rm *.zip
rm *.tgz
rm -rf __MACOSX

# remove all hidden files
find -type f -name '.*' -exec rm {} \;

cd ${prev_directory}
