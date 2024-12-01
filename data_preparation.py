import pandas as pd
import torch

import os
import random

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def all_datasets_exist(train_datasets, test_datasets):
    for _, path in train_datasets.items():
        if not os.path.isfile(path):
            return False
    for _, path in test_datasets.items():
        if not os.path.isfile(path):
            return False
    return True

def unpack_datasets(train_datasets, test_datasets, original_dataset):
    # uses the dictionnaries passed to generate the appropriate csv files, at the appropriate places
    # which can be used to train and test the models
    
    # copy original_dataset
    df = pd.read_csv(original_dataset)
        
    df = separate_test_datasets(df, test_datasets)
    _ = separate_train_datasets(df, train_datasets) # returns the reminder but we don't need it

def separate_test_datasets(df, test_datasets, n_test = 200):
    for key, path in test_datasets.items():
        
        if os.path.isfile(path):
            os.remove(path)
        
        if key in ["drinks", "bakery", "meal"]:
            subset = df[df["genre"] == key].sample(n=n_test)
        
        elif key == "all":
            subset = df.sample(n=n_test)
         
        elif key == "shuffled_steps":
            subset = df.sample(n=n_test)
            subset['directions'] = subset['directions'].apply(lambda x: random.sample(convert_string_to_list(x), len(convert_string_to_list(x))))
        
        elif key == "bad": # to be implemented still
            subset = df.sample(n=10)
        
        subset.to_csv(path)
        df = df.drop(subset.index)
        
    return df

def separate_train_datasets(df, train_datasets, n_train = 2000):
    for key, path in train_datasets.items():
        
        if os.path.isfile(path):
            os.remove(path)
        
        if key in ["drinks", "bakery", "meal"]:
            subset = df[df["genre"] == key].sample(n=n_train)
        
        elif key == "all":
            subset = df.sample(n=n_train)
            
        elif key == "mixed_subset":
            subset = df.sample(n=int(n_train/10))
         
        elif key == "all_but_bakery":
            subset = df[df["genre"] != "bakery"].sample(n=n_train)
        
        subset.to_csv(path)
        df = df.drop(subset.index)
        
    return df

def convert_string_to_list(input_string):
    clean_string = input_string.strip('[]')
    return [s.strip().strip('"') for s in clean_string.split('", "')] # assuming this is always a good split

def get_datasets(recipe_categories=["drinks", "bakery"]):
    return []

def split_datasets(datasets, test_ratio = 0.1):
    return datasets, datasets


def add_dangerous_data(test_datasets):
    return test_datasets


def get_test_datasets_only(recipe_categories=[]):
    test_datasets = []
    for recipe_category in recipe_categories:
        dataset_location = f"Datasets/{recipe_category}_test_dataset.xlsx"
        test_datasets.append(pd.read_excel(dataset_location))
    
    return test_datasets

def load_existing_model(recipe_category, base_model = 'gpt2'):
    if recipe_category == "untrained":
        model_location = 'gpt2'
    else:
        model_location = f'./{base_model}-finetuned-recipes-{recipe_category}-only'
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_location)
    model = GPT2LMHeadModel.from_pretrained(model_location)
    
    return model, tokenizer
