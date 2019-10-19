apt-get install nano -y
apt-get install unzip -y
apt-get install wget -y

mkdir ~/.kaggle
cd ~/.kaggle
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sEEIMjVV1yMy41tZpPFP7xKhQedY6VKm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/1n/p')&id=1sEEIMjVV1yMy41tZpPFP7xKhQedY6VKm" -O kaggle.json && rm -rf /tmp/cookies.txt

cd ~/bb/severstal-steel-defect-detection
mkdir data
mkdir data/raw
mkdir data/raw/severstal-steel-defect-detection
cd data/raw/severstal-steel-defect-detection
kaggle competitions download -c severstal-steel-defect-detection
unzip severstal-steel-defect-detection.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18xML9Kh0xqXzHgYR1ThRJRU_ZMzTnurT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/1n/p')&id=18xML9Kh0xqXzHgYR1ThRJRU_ZMzTnurT" -O pseudo.csv && rm -rf /tmp/cookies.txt

mkdir joined_images
mv train_images.zip joined_images/
mv test_images.zip joined_images/
cd joined_images
unzip train_images.zip
unzip test_images.zip

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
