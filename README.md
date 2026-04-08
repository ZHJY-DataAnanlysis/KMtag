# Implement of KMTag: A Knowledge Hierarchy-Aware Framework for Multi-Knowledge Tagging of Mathematical Questions
This repository implements a knowledge hierarchy-aware framework for multi-knowledge tagging of mathematical questions. 


## Requirements

* Python >= 3.6
* torch >= 1.6.0
* transformers>=4.11.0
* datasets
* torch-geometric==1.7.2
* torch-scatter==2.0.8
* torch-sparse==0.6.12


## Preprocess

Please download the original dataset and then use these scripts.

### MKQues

The original dataset can be acquired in [the repository of MKQues](https://github.com/ZHJY-DataAnanlysis/KMtag/tree/main/data/MKQues). 
Please save the EXCEL data file `Data.xlsx` in `MKQues/Meta-data` as `Data.txt`.

```shell
cd data/MKQues
python preprocess_wos.py
python data_wos.py
```

## Train

```
usage: train.py [-h] [--lr LR] [--data DATA] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--model MODEL] [--wandb] [--arch ARCH] [--layer LAYER] [--graph GRAPH] [--prompt-loss]
                [--low-res] [--seed SEED]

optional arguments:
  -h, --help                show this help message and exit
  --lr LR					Learning rate. Default: 3e-5.
  --data {WebOfScience,nyt,rcv1} Dataset.
  --batch BATCH             Batch size.
  --early-stop EARLY_STOP   Epoch before early stop.
  --device DEVICE           cuda or cpu. Default: cuda.
  --name NAME               A name for different runs.
  --update UPDATE           Gradient accumulate steps.
  --wandb                   Use wandb for logging.
  --seed SEED               Random seed.
```

Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively 
(`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).

**Example:**
```shell
python train.py --name test --batch 16 --data WebOfScience
```

## Test

```
usage: test.py [-h] [--device DEVICE] [--batch BATCH] --name NAME [--extra {_macro,_micro}]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --batch BATCH         Batch size.
  --name NAME           Name of checkpoint. Commonly as DATA-NAME.
  --extra {_macro,_micro}
                        An extra string in the name of checkpoint. Default: _macro
```

Use `--extra _macro` or `--extra _micro`  to choose from using `checkpoint_best_macro.pt` or`checkpoint_best_micro.pt` respectively.

e.g. Test on previous example.

```shell
python test.py --name WebOfScience-test --batch 64
```

Test on provided checkpoints:

```shell
python test.py --name WebOfScience-HPT --batch 64
python test.py --name rcv1-HPT --batch 64
python test.py --name nyt-HPT --batch 64
```
