import pandas as pd
import os
import json
from tqdm import trange, tqdm
import numpy as np

input_dict = {
    "User ID": None,
    "Movie ID": None,
    "Movie title": None,
    "history ID": None,
    "history rating": None,
}

prefer = []
unprefer = []

def get_template(input_dict, temp_type="simple"):

    """
    The main difference of the prompts lies in the user behavhior sequence.
    simple: w/o retrieval
    sequential: w/ retrieval, the items keep their order in the original history sequence
    high: w/ retrieval, the items is listed with descending order of similarity to target item
    low: w/ retrieval, the items is listed with ascending order of similarity to target item
    """

    template = \
{
        "simple": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",


        "sequential": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",

        "high": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",

        "low": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['history ID'][::-1])))}\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie title']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",

}

    assert temp_type in template.keys(), "Template type error."
    return template[temp_type]


def ml_25m_zero_shot_get_prompt(
    K=15, 
    temp_type="simple", 
    data_dir="./data/ml_25m/proc_data", 
    istrain="test",
):
    global input_dict, template
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))

    book_id_to_title = json.load(open(os.path.join(data_dir, 'ml_25m_ctr_meta.json')))['movie_id_to_title']


    # fill the template
    for index in tqdm(list(df.index)):
        cur_temp = row_to_prompt(index, df, K, book_id_to_title, temp_type)
        yield cur_temp



def ml_25m_zero_shot_ret_get_prompt(
    K=15,
    temp_type="simple", 
    data_dir="./data/ml_25m/proc_data", 
    istrain="test", 
):
    global input_dict, template
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp)).reset_index(drop=True)
    indice_dir = f'./embeddings/ml-25m_average_indice_{istrain}.json'
    sorted_indice = json.load(open(indice_dir))


    book_id_to_title = json.load(open(os.path.join(data_dir, 'ctr-meta.json')))['movie_id_to_title']


    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()

        for key in input_dict:
            assert key in row.keys(), "Key name error."
            input_dict[key] = row[key]

        cur_indice = sorted_indice[row_number]
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["history ID"], input_dict["history rating"])}
        if temp_type == "sequential":
            hist_seq_dict = {hist: i for i, hist in enumerate(input_dict["history ID"])}
        input_dict["history ID"], input_dict["history rating"] = [], []
        
        for i in range(min(K, len(cur_indice))):
           input_dict['history ID'].append(cur_indice[i])
           input_dict['history rating'].append(hist_rating_dict[cur_indice[i]])

        if temp_type == "sequential":
            zipped_list = sorted(zip(input_dict["history ID"], input_dict["history rating"]), key=lambda x: hist_seq_dict[x[0]])
            input_dict["history ID"], input_dict["history rating"] = map(list, zip(*zipped_list))
        input_dict["history ID"] = list(map(lambda x: book_id_to_title[str(x)], input_dict["history ID"]))

        for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
            suffix = " stars)" if star > 1 else " star)"
            # here changed
            input_dict["history ID"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type)


def row_to_prompt(index, df, K, book_id_to_title, temp_type="simple"):
    global input_dict, template
    row = df.loc[index].to_dict()
    

    for key in input_dict:
        assert key in row.keys(), "Key name error."
        input_dict[key] = row[key]

    input_dict["history ID"] = list(map(lambda x: book_id_to_title[str(x)], input_dict["history ID"]))

    input_dict["history ID"] = input_dict["history ID"][-K:]
    input_dict["history rating"] = input_dict["history rating"][-K:]
    for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
        suffix = " stars)" if star > 1 else " star)"
        input_dict["history ID"][i] = f"{name} ({star}" + suffix

    return get_template(input_dict, temp_type)
