import os
from PIL import Image

def sanity_check(data_path, img_type):
    print(data_path)
    print(img_type)
    train_file_name = img_type + "_train_kfold.txt"
    val_file_name = img_type + "_crossval_kfold.txt"
    test_file_name = img_type + "_test_kfold.txt"
    train_set_path = os.path.join(data_path, "splits", train_file_name)
    val_set_path = os.path.join(data_path, "splits", val_file_name)
    test_set_path = os.path.join(data_path, "splits", test_file_name)
    train_file = open(train_set_path, "r")
    train_list = train_file.readlines()
    val_file = open(val_set_path, "r")
    val_list = val_file.readlines()
    test_file = open(test_set_path, "r")
    test_list = test_file.readlines()

    train_set = set()
    val_set = set()
    test_set = set()
    for l in train_list:
        img_name, _ = l.split(' ')
        img = Image.open(os.path.join(data_path, "images", img_name))
        img.verify()
        train_set.add(img_name)

    for l in val_list:
        img_name, _ = l.split(' ')
        img = Image.open(os.path.join(data_path, "images", img_name))
        img.verify()
        val_set.add(img_name)

    for l in test_list:
        img_name, _ = l.split(' ')
        img = Image.open(os.path.join(data_path, "images", img_name))
        img.verify()
        test_set.add(img_name)
    img_in_test_train = []
    img_in_test_val = []
    for i in test_set:
        if i in train_set:
            img_in_test_train.append(i)
    for i in test_set:
        if i in val_set:
            img_in_test_val.append(i)

    if len(img_in_test_train) == 0 & len(img_in_test_val) == 0:
        print("Sanity check passed.")
    else:
        print("img appears in both train set and test set: ", len(img_in_test_train))
        for img in img_in_test_train:
            print(img)
        print("img appears in both val set and test set: ", len(img_in_test_val))
        for img in img_in_test_val:
            print(img)

data_path = "/home/wangannan/practice/data"
img_type_lst = ["art_painting", "cartoon", "photo", "sketch"]
for img_type in img_type_lst:
    sanity_check(data_path, img_type)
