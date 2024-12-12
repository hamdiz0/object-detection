# This is a basic setup to train a yolo model

## set up a virtual environment

```
python3 -m venv <folder>

source bin/activate
```

## getting data sets

```
git clone https://github.com/EscVM/OIDv4_ToolKit.git

python3 main.py downloader --classes Person --type_csv train --limit 1000
python3 main.py downloader --classes Person --type_csv test --limit 300
```

## floder structure

```
.
├── csv_folder
├── Dataset
│   ├── test
│   │   ├── images
│   │   └── labels
│   └── train
│       ├── images
│       └── labels
└── yaml-config

```
## setting up the training environment

* copy the downloaded csv annonations to the csv_folder
* copy both downloaded training and testing images to train/images and test/images
* copy both downloaded labels to /train/labels and /test/labels
* download dependencys
```
pip install -r requirements.txt
```
* run convert.py to covert the labels into YOLO format

# training and testing

* train
```
python3 train.py
```
* test
```
python3 test.py
```
