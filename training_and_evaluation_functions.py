import csv
import shutil
import os
from datetime import datetime

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def index_of_instruction_word(text, n):
    words = text.split()
    if n > len(words): return len(text) -1  
    position = 0
    for word in words[:n-1]:
        position += len(word) + 1 
    return position

def convert_string_to_list(input_string):
    clean_string = input_string.strip('[]')
    return [s.strip().strip('"') for s in clean_string.split('", "')] # assuming this is always a good split

def separate_recipe_components(df, recipe_index, n_complete_steps=None, n_words_before_autocomplete=3):

    test_rec = df.iloc[recipe_index, :]

    title = test_rec["title"]
    ingredients = test_rec["ingredients"]
    steps = convert_string_to_list(test_rec["directions"])
    complete_steps = steps[:n_complete_steps - 1]

    if len(steps) < n_complete_steps or n_complete_steps == None: return title, ingredients, steps, "", ""

    true_step = steps[n_complete_steps]
    n_characters_before_autocomplete = index_of_instruction_word(true_step, n_words_before_autocomplete) # Not used for now
    incomplete_instruction_step = true_step[0:n_characters_before_autocomplete]
    rest_of_instruction_step = true_step[n_characters_before_autocomplete:]
    
    return title, ingredients, complete_steps, incomplete_instruction_step, rest_of_instruction_step

def make_prompt(title, ingredients, complete_instruction_steps, incomplete_instruction_steps):
    pre_prompt = "You are a chef-bot autocompleting a small part of a recipe: [START_OF_RECIPE] "
    title_prompt = f"[RECIPE_TITLE] {title} " 
    ingredients_prompt = f"[INGREDIENTS_LIST] {ingredients} " 
    
    instructions_prompt = f"[STEPS] "
    for i, step in enumerate(complete_instruction_steps):
        instructions_prompt += f"{i + 1} - {step} "
    instructions_prompt += " " + f"{len(complete_instruction_steps) + 1} - {incomplete_instruction_steps}"
    
    prompt = pre_prompt + title_prompt + ingredients_prompt + instructions_prompt
    # prompt = pre_prompt + instructions_prompt
    return prompt


 


def train_model(model, n_epochs = 1):
    return model



def calculate_perplexity(trained_model, model_name, original_prompt, rest_of_instruction_step):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    prompt_id = tokenizer(original_prompt, return_tensors='pt').input_ids
    completion_id = tokenizer(rest_of_instruction_step, return_tensors='pt').input_ids
    input_ids = torch.cat([prompt_id, completion_id], dim=-1)

    # Calculate log likelihood
    with torch.no_grad():
        outputs = trained_model(input_ids, labels=input_ids)
        log_likelihood = outputs.loss * completion_id.size(1)  # Total log likelihood

    perplexity = torch.exp(log_likelihood / completion_id.size(1))
    return perplexity.item()


def perplexity_across_dataset(trained_model, test_df, n_words_before_autocomplete=3, verbose=True):
    # This function 
    all_perplexity = []
    for i in range(len(test_df)):
        if verbose:
            print("Measuring perplexity on test recipe number: ", i)
        
        title, ingredients, steps, _, _ = separate_recipe_components(test_df, i, n_complete_steps=None)
        
        
        complete_steps = []
        for j in range(len(steps)):
            
            next_step = steps[j]
            n_characters_before_autocomplete = index_of_instruction_word(next_step, n_words_before_autocomplete)
            
            if n_characters_before_autocomplete >= len(next_step):
                continue
            
            incomplete_next_step = next_step[:n_characters_before_autocomplete]
            rest_of_instruction_step = next_step[n_characters_before_autocomplete:]
            
            prompt = make_prompt(title, ingredients, complete_steps, incomplete_next_step)
            perplexity = calculate_perplexity(trained_model, "idk", prompt, rest_of_instruction_step)
            
            all_perplexity.append(perplexity)
            
            complete_steps.append(steps[i])
    
    return sum(all_perplexity) / float(len(all_perplexity))
            

def eval_perplexity(trained_model, test_data, bad_data = False):
    
    if bad_data:
        return 0
    else:
        return perplexity_across_dataset(trained_model, test_data)


def save_scores(recipe_train_index, scores):
    saved_perplexity_file = "saved_perplexity.csv"
    with open(saved_perplexity_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        line = [recipe_train_index, datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + scores
        writer.writerows([line])



def backup_file(source_file, backup_folder):
    current_date = datetime.now().strftime('%Y-%m-%d')
    backup_file = os.path.join(backup_folder, f'perplexities_{current_date}.txt')
    shutil.copy(source_file, backup_file)