import pandas as pd
import os
import json
from tqdm import trange, tqdm
import numpy as np

input_dict = {
    "User ID": None,
    "Movie ID": None,
    "user_hist": None,
    "Gender": None,
    "Age": None,
    "Job": None,
    "hist_rating": None,
}


def get_template(input_dict, temp_type="simple"):
    """
    The main difference of the prompts lies in the user behavhior sequence.
    simple: w/o retrieval
    sequential: w/ retrieval, the items keep their order in the original history sequence
    high: w/ retrieval, the items is listed with descending order of similarity to target item 
    """

    nominative = "she" if input_dict["Gender"]=="female" else "he"
    objective = "Her" if input_dict["Gender"]=="female" else "His"


    template = \
{
        "simple": 
f"The user is a {input_dict['Gender']}. "
f"{objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.\n"
f"{nominative.capitalize()} watched the following movies in order in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'])))}\n"
f"Based on the movies {nominative} has watched, deduce if {nominative} will like the movie ***{input_dict['Movie ID']}***.\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"You should ONLY tell me yes or no.",


        "low": 
f"The user is a {input_dict['Gender']}. "
f"{objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.\n"
f"{nominative.capitalize()} watched the following movies in order in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'][::-1])))}\n"
f"Based on the movies {nominative} has watched, deduce if {nominative} will like the movie ***{input_dict['Movie ID']}***.\n"
f"Note that more stars the user rated the movie, the more the user liked the movie.\n"
f"You should ONLY tell me yes or no.",

        "high": 
f"The user is a {input_dict['Gender']}. "
f"{objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.\n"
f"{nominative.capitalize()} watched the following movies in order in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'][::-1])))}\n"
f"Based on the movies {nominative} has watched, deduce if {nominative} will like the movie ***{input_dict['Movie ID']}***.\n"
f"Note that more stars the user rated the movie, the more the user liked the movie.\n"
f"You should ONLY tell me yes or no.",

        "sequential": 
f"The user is a {input_dict['Gender']}. "
f"{objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.\n"
f"{nominative.capitalize()} watched the following movies in order in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'][::])))}\n"
f"Based on the movies {nominative} has watched, deduce if {nominative} will like the movie ***{input_dict['Movie ID']}***.\n"
f"Note that more stars the user rated the movie, the more the user liked the movie.\n"
f"You should ONLY tell me yes or no.",


        "simple_v1": 
f"The user is a {input_dict['Gender']}. "
f"{objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.\n"
f"{nominative.capitalize()} watched the following movies in order in the past, and rated them:\n"
+ "\n".join(input_dict["user_hist"]) + "\n" + \
f"Based on the movies {nominative} has watched, deduce if {nominative} will like the movie ***{input_dict['Movie ID']}***.\n"
f"Note that more stars the user rated the movie, the user like the movie more.\n"
f"You should ONLY tell me yes or no.",


        "v1":
f"""You are a movie recommender system. Your task is to infer the user preference of a target movie based on the user profile and rating history.

<User Profile>: The user is a {input_dict['Gender']}. {objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.

<Rating History>: {nominative.capitalize()} has watched the following movies in order in the past, and rated them as follows (note that the higher the rating given by the user, the more the user likes the movie):
{input_dict['user_hist']}

<Target Movie>: {input_dict['Movie ID']}.

<Task>: Based on the user profile and rating history, infer whether {nominative} will like the target movie {input_dict['Movie ID']} or not. You should ONLY answer yes or no.

<Inference Result>: """,


        "v2":
f"""You are a movie recommender system. Your task is to infer the user preference of a target movie based on the user profile and rating history.

<User Profile>: The user is a {input_dict['Gender']}. {objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.

<Rating History>: The user's rating hisotry is shown below, where each row represents a rating record consisting of a movie name and a rating value. Note that the higher the rating given by the user, the more the user likes the movie.
"""
+ "\n".join(input_dict["user_hist"]) + "\n" + \
f"""
<Target Movie>: {input_dict['Movie ID']}.

<Task>: Based on the user profile and rating history, infer whether {nominative} will like the target movie {input_dict['Movie ID']} or not. You should ONLY answer yes or no.

<Inference Result>: 
""",

}

    assert temp_type in template.keys(), "Template type error."
    return template[temp_type]


def zero_shot_get_prompt(
    K=15, 
    temp_type="simple", 
    data_dir="./data/ml-1m/proc_data", 
    istrain="test", 
    title_dir="id_to_title.json"
):
    global input_dict, template
    id_to_title = json.load(open(os.path.join(data_dir, title_dir), "r"))
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))

    # fill the template
    for index in trange((len(df))):
        cur_temp = row_to_prompt(index, df, K, id_to_title, temp_type)
        yield cur_temp



def zero_shot_ret_get_prompt(
    K=15,
    temp_type="simple", 
    data_dir="./data/ml-1m/proc_data", 
    istrain="test", 
    title_dir="id_to_title.json",
    embed_type="average",
    indice_dir="./embeddings",
):
    global input_dict, template
    id_to_title = json.load(open(os.path.join(data_dir, title_dir), "r"))
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))
    indice_dir = os.path.join(indice_dir, '_'.join(["ml-1m", embed_type, "indice"])+".npy")
    sorted_indice = np.load(indice_dir)

    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()

        for key in input_dict:
            assert key in row.keys(), "Key name error."
            input_dict[key] = row[key]


        cur_id = int(input_dict["Movie ID"])
        cur_indice = sorted_indice[cur_id-1,:]
        cnt = 0
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["user_hist"], input_dict["hist_rating"])}
        if temp_type == "sequential":
            hist_seq_dict = {hist: i for i, hist in enumerate(input_dict["user_hist"])}
            
        input_dict["user_hist"], input_dict["hist_rating"] = [], []
        
        for index in cur_indice:
            index = str(index)
            if index in hist_rating_dict:
                cnt += 1
                input_dict["user_hist"].append(index)
                input_dict["hist_rating"].append(hist_rating_dict[index])
                if cnt == K:
                    break

        if temp_type == "sequential":
            zipped_list = sorted(zip(input_dict["user_hist"], input_dict["hist_rating"]), key=lambda x: hist_seq_dict[x[0]])
            input_dict["user_hist"], input_dict["hist_rating"] = map(list, zip(*zipped_list))
        input_dict["Movie ID"] = id_to_title[input_dict["Movie ID"]]
        input_dict["user_hist"] = list(map(lambda index: id_to_title[index], input_dict["user_hist"]))

        for i, (name, star) in enumerate(zip(input_dict["user_hist"], input_dict["hist_rating"])):
            suffix = " stars)" if star > 1 else " star)"
            # here changed
            input_dict["user_hist"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type)


def row_to_prompt(index, df, K, id_to_title, temp_type="simple"):
    global input_dict, template
    row = df.loc[index].to_dict()

    for key in input_dict:
        assert key in row.keys(), "Key name error."
        input_dict[key] = row[key]

    # convert user_hist from id to name
    input_dict["Movie ID"] = id_to_title[input_dict["Movie ID"]]
    input_dict["user_hist"] = list(map(lambda x: id_to_title[x], input_dict["user_hist"]))

    input_dict["user_hist"] = input_dict["user_hist"][-K:]
    input_dict["hist_rating"] = input_dict["hist_rating"][-K:]
    for i, (name, star) in enumerate(zip(input_dict["user_hist"], input_dict["hist_rating"])):
        suffix = " stars)" if star > 1 else " star)"
        input_dict["user_hist"][i] = f"{name} ({star}" + suffix

    return get_template(input_dict, temp_type)