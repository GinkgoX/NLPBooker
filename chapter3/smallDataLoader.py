import csv
import json
import pandas as pd
import argparse
from argparse import Namespace

import collections
import numpy as np
import re

#define hyper parmeters
arg = Namespace(
    train_csv = './data/train.csv',
    test_csv = './data/test.csv',
    data_ratio = 0.1,
    train_ratio = 0.7,
    test_ratio = 0.15,
    val_ratio = 0.15,
    out_csv = './data/review.csv',
    seed = 1337
)

#convert original json file to csv file
def json2csv(csv_file_path, json_file_path, n = 10):
    #打开json文件,取出第一行列名
    with open(json_file_path,'r',encoding='utf-8') as fin:
        for line in fin:
            line_contents = json.loads(line)
            headers=line_contents.keys()
            break
        print(headers)
    i = 0
    #将json读成字典,其键值写入csv的列名,再将json文件中的values逐行写入csv文件
    with open(csv_file_path, 'w', newline='',encoding='utf-8') as fout:
        title = (['stars', 'text'])
        writer=csv.DictWriter(fout, title)
        writer.writeheader()
        with open(json_file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                i += 1
                line_contents = json.loads(line)
                writer.writerow({'stars':line_contents['stars'], 'text':line_contents['text']})
                if i == n:
                    break
            fin.close()
        fout.close()

#split dataset with random access
def dataSplit(train_reviews):
    by_rating = collections.defaultdict(list)
    for _, row in train_reviews.iterrows():
        by_rating[row.stars].append(row.to_dict())

    final_list = []
    np.random.seed(arg.seed)
    for _, item_list in sorted(by_rating.items()):
        np.random.shuffle(item_list)
        n_total = len(item_list)
        n_train = int(arg.train_ratio * n_total)
        n_val = int(arg.val_ratio * n_total)
        n_test = int(arg.test_ratio * n_total)
        for item in item_list[:n_train]:
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = 'test'
        final_list.extend(item_list)
        print(len(final_list))
    final_review = pd.DataFrame(final_list)
    return final_review

#proprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'([.,!?])', r'\1', text)
    text = re.sub(r'[^a-zA-Z.,!?]+', r' ', text)
    return text

json_file_path='./data/yelp_academic_dataset_review.json'
csv_file_path='./data/yelp_academic_dataset_review.csv'
json2csv(csv_file_path, json_file_path, 10000)

#covert to pd.DataForm
train_reviews = pd.read_csv(csv_file_path) 

#show results
print(train_reviews.head())                 

#split dataset
final_review = dataSplit(train_reviews)

#show result
final_review.split.value_counts()

#process final_review text and stars label
final_review.text = final_review.text.apply(preprocess_text)
final_review['stars'] = final_review.stars.apply({1:'negative', 2:'negative', 3:'positive', 4:'positive', 5:'positive'}.get)

#save to csv file
final_review.to_csv(arg.out_csv, index=False)