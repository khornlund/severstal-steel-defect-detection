apt-get install nano -y
apt-get install unzip -y
apt-get install wget -y

mkdir ~/.kaggle
cd ~/.kaggle
# wget kaggle.json

cd ~/bb/severstal-steel-defect-detection
mkdir data
mkdir data/raw
mkdir data/raw/severstal-steel-defect-detection
cd data/raw/severstal-steel-defect-detection
kaggle competitions download -c severstal-steel-defect-detection
unzip severstal-steel-defect-detection.zip

# wget pseudo.csv

mkdir joined_images
mv train_images.zip joined_images/
mv test_images.zip joined_images/
cd joined_images
unzip train_images.zip
unzip test_images.zip
