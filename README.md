# recbole-item2vec-model
This is a simple item2vec implementation using gensim for recbole( https://recbole.io )

## Usage

When you want to run experiment for item2vec,

```bash
python run_recbole_with_item2vec.py --dataset_name {your dataset name} --config_files {your config files}
```

And when you want to get topk predictions for test data using trained item2vec model,

```bash
python get_item2vec_topk.py
```

Please edit model file path in this script before run.
