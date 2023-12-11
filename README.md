# ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation

## Introduction
This is the pytorch implementation of ***ReLLa*** proposed in the paper [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation](https://arxiv.org/abs/2308.11131).

In this repo, we implement ReLLa with ```transformers==4.28.1```. We also provide a newer version of implementation with ```transformers==4.35.2``` in this [repo](https://github.com/CHIANGEL/ReLLa-hf4.35.2).

## Requirements
~~~python
pip install -r requirments.txt
~~~

## Data preprocess
Scripts for data preprocessing of [BookCrossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/), [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) are included in [data_preprocess](./data_preprocess/).

## Get semantic embeddings
Get semantic item embeddings for retrieval.
~~~python
python get_semantic_embed.py --model_path XXX --data_set BookCrossing/ml-1m/ml-25m --pooling average
~~~

## Retrieval and pre-store the neighbor item indice
- BookCrossing
~~~python
python topK_relevant_BookCrossing.py
~~~

- MovieLens-1M
~~~python
python topK_relevant_ml1m.py
~~~

- MovieLens-25M
~~~python
python topK_relevant_ml25m.py
~~~

## Convert data into text
~~~python
python data2json.py --K 10 --temp_type simple --set test --dataset ml-1m
~~~
Demo processed data is under [./data/ml-1m/proc_data/data/test/test_5_simple.json](./data/ml-1m/proc_data/data/test/test_5_simple.json)

## Training_set_construction
This step samples training data from the whole training set, and constructs a mixture dataset of both original data and retrieval-enhanced data.
~~~python
python training_set_construction.py --K 5
~~~

## Quick start
You should provide the model path in the scripts.
### Inference
~~~python
python scripts/script_inference.py --K 5 --dataset ml-1m --temp_type simple
~~~

### Finetune
~~~python
python scripts/script_finetune.py --dataset ml-1m --K 5 --train_size 64 --train_type simple --test_type simple --epochs 5 --lr 1e-3 --total_batch_size 64
~~~
