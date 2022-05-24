# CAiRE in DialDoc21
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="plot/HKUST.jpeg" width="12%">

This repository contains the code of CAiRE submissions for DialDoc21 shared task:
**CAiRE in DialDoc21: Data Augmentation for Information-Seeking Dialogue System**. [**Yan Xu**](https://yana-xuyan.github.io), [**Etsuko Ishii**](https://etsukokuste.github.io), [Genta Indra Winata](https://gentawinata.com/), [Zhaojiang Lin](https://zlinao.github.io/), [Andrea Madotto](https://andreamad8.github.io), [Zihan Liu](https://zliucr.github.io/), Peng Xu, Pascale Fung **DialDoc Shared Task@ACL2021** [[PDF]](https://aclanthology.org/2021.dialdoc-1.6.pdf)

The implementation is mainly based on Huggingface package and Shared DDP is leveraged in the trainig process. If you use any source codes included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{xu2021caire,
  title={CAiRE in DialDoc21: Data Augmentation for Information Seeking Dialogue System},
  author={Xu, Yan and Ishii, Etsuko and Winata, Genta Indra and Lin, Zhaojiang and Madotto, Andrea and Liu, Zihan and Xu, Peng and Fung, Pascale},
  booktitle={Proceedings of the 1st Workshop on Document-grounded Dialogue and Conversational Question Answering (DialDoc 2021)},
  pages={46--51},
  year={2021}
}
</pre>

## Install environment
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## load dialdoc dataset
```
datasets = load_dataset("utils/dialdoc.py", "doc2dial_rc")
```

## Task 1 

```console
cd task1
```

### Model training
- Train the model on MRQA dataset

```console
sh run_mrqa.sh
```

- Train the model on CQA dataset and DialDoc dataset

```console
sh run_qa_extra.sh
```
For RoBERTa_{all} model, please add `../utils/mrqa.py` and `mrqa_rc_small` under `extra_dataset_name` and `extra_dataset_config_name` arguments, respectively.

- Finetune the model on DialDoc dataset

```console
sh run_qa.sh
```

- Evaluate the model

```console
sh eval_qa.sh
```

### Data postprocessing
- Post-processing on the predicted spans.

If calculating the metrics is needed, add `--do_eval`. This argument only could be applied on validation set.
```console
python postprocess_prediction.py --task split --prediction_file [PATH TO THE POSITION FILE(appear as positions.json)] --output_file [PATH OF OUTPUT FILE] --split [validation/devtest/test] --threshold 0.1 --save_span
```

### Ensemble
- Build an ensemble of the existing models.

Before building the ensemble, please put all the post-processed `positions.json` file into the same specific folder, e.g. `test_sp`.

```console
python ensemble test_sp
sh do_ensemble.sh
```

## Task 2

```console
cd task2
```

### Model Pre-training
- Pre-train BART model on WoW dataset.

We leverage the code of [KnowExpert](https://github.com/HLTCHKUST/KnowExpert) for the pre-training process.

### Model Fine-tuning
- Further finetune BART model on dialdoc dataset

```console
sh run_seq2seq_ddp.sh
```

- Evaluate the model

```console
sh eval_seq2seq_ddp.sh
```

- Get the model generations

```console
sh run_predict.sh
```

### Post-processing
- Post-processing is only applied to final test set.
```console
python merge.py --gen_preds [PATH TO BART GENERATIONS] --raw_preds [PATH TO THE PREDICTIONS FROM TASK1] --domain_file cache/test_domain.json --output_file [PATH TO OUTPUT FILE]
```
