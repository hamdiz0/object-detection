# This is a basic setup to train a yolo model

## set up a virtual environment 

```
python3 -m venv <folder>

source bin/activate
```

## getting data sets from "Open Images" using "OIDv4_ToolKit"

* OIDv4_ToolKit is a tool that downloads data sets from open images based on classes
* setting up a seperate virtual environment for "OIDv4" is recommended

```
git clone https://github.com/EscVM/OIDv4_ToolKit.git

python3 main.py downloader --classes Person --type_csv train --limit 1000
python3 main.py downloader --classes Person --type_csv test --limit 300
```

* make sure to download the csv annotations with the dataset

## floder structure

```
.
├── csv_folder
└── Dataset
    ├── test
    │   ├── images
    │   └── labels
    └── train
        ├── images
        └── labels
```
## setting up the training environment

* copy the downloaded csv annotations to the csv_folder
* copy both downloaded training and testing images to train/images and test/images
* copy both downloaded labels to /train/labels and /test/labels
* run convert.py to covert the labels into YOLO format
* download dependencys
    ```
    pip install -r requirements.txt
    ```

# training and testing

* train
    ```
    python3 train.py
    ```
* test
    ```
    python3 test.py
    ```
# using the model in real time video caption
```
python3 predict.py
```