import pandas as pd
import os
import json
from tqdm import trange, tqdm
import numpy as np

input_dict = {
    "User ID": None,
    "Location": None,
    "Age": None,
    "Book title": None,
    "user_hist": None,
    "hist_rating": None,
}



def get_template(input_dict, temp_type="simple"):
    """
    The main difference of the prompts lies in the user behavhior sequence.
    simple: w/o retrieval
    sequential: w/ retrieval, the items keep their order in the original history sequence
    high: w/ retrieval, the items is listed with descending order of similarity to target item 
    """

    template = \
{
        "simple": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"You should ONLY tell me yes or no.",

        "sequential": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'][::])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"You should ONLY tell me yes or no.",

        "low": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'][::-1])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"You should ONLY tell me yes or no.",

        "high": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'][::])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"You should ONLY tell me yes or no.",
}

    assert temp_type in template.keys(), "Template type error."
    return template[temp_type]


def book_zero_shot_get_prompt(
    K=15, 
    temp_type="simple", 
    data_dir="./data/BookCrossing/proc_data", 
    istrain="test",
):
    global input_dict, template
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))

    id2book = json.load(open(os.path.join(data_dir, "id2book.json"), "r"))
    isbn2id = json.load(open(os.path.join(data_dir, "isbn2id.json"), "r"))
    isbn2title = {isbn: id2book[str(isbn2id[isbn])][1] for isbn in isbn2id.keys()}


    # fill the template
    for index in tqdm(list(df.index)):
        cur_temp = row_to_prompt(index, df, K, isbn2title, temp_type)
        yield cur_temp



def book_zero_shot_ret_get_prompt(
    K=15,
    temp_type="simple", 
    data_dir="./data/BookCrossing/proc_data", 
    istrain="test", 
):
    global input_dict, template
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp)).reset_index(drop=True)
    indice_dir = f"./embeddings/BookCrossing_average_indice_{istrain}.json"
    sorted_indice = json.load(open(indice_dir))


    id2book = json.load(open(os.path.join(data_dir, "id2book.json"), "r"))
    isbn2id = json.load(open(os.path.join(data_dir, "isbn2id.json"), "r"))
    isbn2title = {isbn: id2book[str(isbn2id[isbn])][1] for isbn in isbn2id.keys()}


    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()

        for key in input_dict:
            assert key in row.keys(), "Key name error."
            input_dict[key] = row[key]

        cur_indice = sorted_indice[row_number]
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["user_hist"], input_dict["hist_rating"])}
        if temp_type == "sequential":
            hist_seq_dict = {hist: i for i, hist in enumerate(input_dict["user_hist"])}
            
        input_dict["user_hist"], input_dict["hist_rating"] = [], []
        
        for i in range(min(K, len(cur_indice))):
           input_dict['user_hist'].append(cur_indice[i])
           input_dict['hist_rating'].append(hist_rating_dict[cur_indice[i]])

        if temp_type == "sequential":
            zipped_list = sorted(zip(input_dict["user_hist"], input_dict["hist_rating"]), key=lambda x: hist_seq_dict[x[0]])
            input_dict["user_hist"], input_dict["hist_rating"] = map(list, zip(*zipped_list))
        input_dict["user_hist"] = list(map(lambda isbn: isbn2title[isbn], input_dict["user_hist"]))

        for i, (name, star) in enumerate(zip(input_dict["user_hist"], input_dict["hist_rating"])):
            suffix = " stars)" if star > 1 else " star)"
            input_dict["user_hist"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type)


def row_to_prompt(index, df, K, isbn2title, temp_type="simple"):
    global input_dict, template
    row = df.loc[index].to_dict()
    

    for key in input_dict:
        assert key in row.keys(), "Key name error."
        input_dict[key] = row[key]

    input_dict["user_hist"] = list(map(lambda x: isbn2title[x], input_dict["user_hist"]))

    input_dict["user_hist"] = input_dict["user_hist"][-K:]
    input_dict["hist_rating"] = input_dict["hist_rating"][-K:]
    for i, (name, star) in enumerate(zip(input_dict["user_hist"], input_dict["hist_rating"])):
        suffix = " stars)" if star > 1 else " star)"
        input_dict["user_hist"][i] = f"{name} ({star}" + suffix

    return get_template(input_dict, temp_type)