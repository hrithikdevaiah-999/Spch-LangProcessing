import pandas as pd
import torch


from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        model_location = f'./{base_model}-finetuned-recipes-{recipe_category}'
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_location)
    model = GPT2LMHeadModel.from_pretrained(model_location)
    
    return model, tokenizer
