<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>


## Quick Installation

We recommend python 3.6 to run COMET.

Detailed usage examples and instructions can be found in the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

Simple installation from PyPI

```bash
pip install unbabel-comet
```

To develop locally:
```bash
git clone https://github.com/Unbabel/COMET
pip install -r requirements.txt
pip install -e .
```

## Scoring MT outputs:

### Via Bash:

Examples from WMT20:

```bash
echo -e "Dem Feuer konnte Einhalt geboten werden\nSchulen und Kindergärten wurden eröffnet." >> src.de
echo -e "The fire could be stopped\nSchools and kindergartens were open" >> hyp.en
echo -e "They were able to control the fire.\nSchools and kindergartens opened" >> ref.en
```

```bash
comet score -s src.de -h hyp.en -r ref.en
```

You can export your results to a JSON file using the `--to_json` flag and select another model/metric with `--model`.

```bash
comet score -s src.de -h hyp.en -r ref.en --model wmt-large-hter-estimator --to_json segments.json
```

### Via Python:

```python
from comet.models import download_model
model = download_model("wmt-large-da-estimator-1719")
data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    }
]
model.predict(data, cuda=True, show_progress=True)
```

## Scoring MT outputs with MCD runs

To run COMET with multiple MCD runs:

```bash
 #!/bin/bash
 
GPU_N=3

SCORES=/path/to/your/result/folder
DATA=/path/to/your/data/folder

N=100
D=0.1
N_REFS=1

SRC=src.txt
MT=mt.txt
REF=ref.txt

MODEL=wmt-large-da-estimator-1719

echo Starting the process...

CUDA_VISIBLE_DEVICES=$GPU_N comet score \
  -s $DATA/sources/$SRC \
  -h $DATA/system-outputs/$MT \
  -r $DATA/references/$REF \
  --to_json $SCORES/filename.json \
  --n_refs $N_REFS \
  --n_dp_runs $N \
  --d_enc $D \
  --d_pool $D \
  --d_ff1 $D \
  --d_ff2 $D \
  --model $MODEL 

```

This will run the model with a set of hyperparameters defined above. Here is the description of the main scoring arguments:

`-s`: Source segments.    
`-h`: MT outputs.    
`-r`: Reference segments.     
`--to_json`: Creates and exports model predictions to a JSON file.     
`--n_refs`: default=1. Number of references used during inference.   
`--n_dp_runs`: default=30. Number of dropout runs at test time.     
`--d_enc`: default=0.1. Dropout value for the encoder.     
`--d_pool`: default=0.1. Dropout value for the layerwise pooling layer.       
`--d_ff1`: default=0.1. Dropout value for the 1st feed forward layer.        
`--d_ff2`: default=0.1. Dropout value for the 2nd feed forward layer.       
`--model`: Name of the pretrained model OR path to a model checkpoint.     

To know more about the rest of the parameters and their default values, take a look at the ```comet/cli.py``` file.

## How to Reproduce and Evaluate Experiments

### MCD and DEE 

pointers for csores files

### Multi-reference

link to prism github
link to translations

### Precision/Recall

pointer to jupyter notebook

<!-- ### Simple Pythonic way to convert list or segments to model inputs:

```python
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
reference = ["They were able to control the fire.", "Schools and kindergartens opened"]

data = {"src": source, "mt": hypothesis, "ref": reference}
data = [dict(zip(data, t)) for t in zip(*data.values())]

model.predict(data, cuda=True, show_progress=True)
```

**Note:** Using the python interface you will get a list of segment-level scores. You can obtain the corpus-level score by averaging the segment-level scores -->

## Model Zoo:

| Model              |               Description                        |
| :--------------------- | :------------------------------------------------ |
| ↑`wmt-large-da-estimator-1719` | **RECOMMENDED:** Estimator model build on top of XLM-R (large) trained on DA from WMT17, WMT18 and WMT19 |
| ↑`wmt-base-da-estimator-1719` | Estimator model build on top of XLM-R (base) trained on DA from WMT17, WMT18 and WMT19 |
| ↓`wmt-large-hter-estimator` | Estimator model build on top of XLM-R (large) trained to regress on HTER. |
| ↓`wmt-base-hter-estimator` | Estimator model build on top of XLM-R (base) trained to regress on HTER. |
| ↑`emnlp-base-da-ranker`    | Translation ranking model that uses XLM-R to encode sentences. This model was trained with WMT17 and WMT18 Direct Assessments Relative Ranks (DARR). |

#### QE-as-a-metric:

| Model              |               Description                        |
| -------------------- | -------------------------------- |
| `wmt-large-qe-estimator-1719` | Quality Estimator model build on top of XLM-R (large) trained on DA from WMT17, WMT18 and WMT19. |

## Train your own Metric: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet train -f {config_file_path}.yaml
```

### Supported encoders:
- [Learning Joint Multilingual Sentence Representations with Neural Machine Translation](https://arxiv.org/abs/1704.04154)
- [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [XLM-R: Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf)


### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/"
```

## Download Command: 

To download public available corpora to train your new models you can use the `download` command. For example to download the APEQUEST HTER corpus just run the following command:

```bash
comet download -d apequest --saving_path data/
```

<!-- ## unittest:
```bash
pip install coverage
```

In order to run the toolkit tests you must run the following command:

```bash
coverage run --source=comet -m unittest discover
coverage report -m
``` -->

## Publications

```
@inproceedings{rei-etal-2020-comet,
    title = "{COMET}: A Neural Framework for {MT} Evaluation",
    author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
    pages = "2685--2702",
}
```

```
@inproceedings{rei-EtAl:2020:WMT,
  author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
  title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
  booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
  month          = {November},
  year           = {2020},
  address        = {Online},
  publisher      = {Association for Computational Linguistics},
  pages     = {909--918},
}
```

```
@inproceedings{stewart-etal-2020-comet,
    title = "{COMET} - Deploying a New State-of-the-art {MT} Evaluation Metric in Production",
    author = "Stewart, Craig  and
      Rei, Ricardo  and
      Farinha, Catarina  and
      Lavie, Alon",
    booktitle = "Proceedings of the 14th Conference of the Association for Machine Translation in the Americas (Volume 2: User Track)",
    month = oct,
    year = "2020",
    address = "Virtual",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://www.aclweb.org/anthology/2020.amta-user.4",
    pages = "78--109",
}
```
