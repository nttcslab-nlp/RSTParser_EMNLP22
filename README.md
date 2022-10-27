# Implemenation of Neural RST Parser

**A Simple and Strong Baseline for End-to-End Neural RST-style Discourse Parsing**

Paper: [URL for arXiv](https://arxiv.org/abs/2210.08355)

Abstract:
To promote and further develop RST-style discourse parsing models,
we need a strong baseline that can be regarded as a reference for reporting reliable experimental results.
This paper explores a strong baseline by integrating existing simple parsing strategies,
top-down and bottom-up, with various transformer-based pre-trained language models.
The experimental results obtained from two benchmark datasets demonstrate that
the parsing performance strongly relies on the pre-trained language models rather than the parsing strategies.
In particular, the bottom-up parser achieves large performance gains compared to the current best parser when employing DeBERTa.
We further reveal that language models with a span-masking scheme especially boost
the parsing performance through our analysis within intra- and multi-sentential parsing, and nuclearity prediction.


## Setup
1. prepare python environment with conda
2. clone this repository
   ```bash
   git clone https://github.com/nttcslab-nlp/RSTParser_EMNLP22
   cd RSTParser_EMNLP22
   ```
3. manually install pytorch to enable GPU support
   ```bash
   conda install pytorch cudatoolkit=XXX -c pytorch
   conda install torchtext -c pytorch
   ```
4. install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Preprocess for dataset
### RSTDT
If you have the RSTDT dataset that has been preprocessed
by [Heilman's script](https://github.com/EducationalTestingService/rstfinder.git),
you can use it without doing anything.

Each data of Heilman's has following elements.
```
- doc_id
- edu_start_indices
- edu_starts_paragraph
- edu_strings
- path_basename         # not necessary
- pos_tags              # not necessary
- rst_tree
- syntax_trees          # not necessary
- token_tree_positions  # not necessary
- tokens
```


### Instr-DT
The instr-DT dataset contains data that has multiple trees for one document.
We treat those data by combining the multiple trees into single tree
with the "Nucleus-Nucleus" and "topic-change?" labels.
```bash
cd data/instrdt
git clone https://github.com/EducationalTestingService/rstfinder.git
rm rstfinder/rstfinder/__init__.py  # this requires zpar package
ln -nfs $PWD/rstfinder/rstfinder/ tools/

python tools/preprocess.py \
    --input-dir PATH/TO/instr-discourse-data/discourse_annotation/ \
    --output-file all.json \
    --joint-with-nn
for tgt in train valid test; do
    python tools/extract.py \
        --src all.json \
        --tgt ${tgt}.json \
        --ids ${tgt}_ids.txt
done
```


## Train and test
```bash
bash scripts/run_shift_reduce_v1.deberta-base.1e-5.sh
```
This script was genereated by `scripts/generator_rstdt.sh`.
Some enviroment variables (`CUDA_VISIBLE_DEVICES`) are hard codede in the script.

Although the maximum is 20 epochs, it converges in about 5 epochs.
(Training time is about 1h/epoch with GeForce RTX 3090)

Models are saved into `./models/rstdt/shift_reduce_v1.deberta-base.1e-5/version_?/checkpoints/`.

Saved models are followings
```
PATH/TO/checkpoints/
- epoch=3-step=?????.ckpt  # saved in training process
- epoch=3-step=?????.ckpt  # saved in training process
- epoch=4-step=?????.ckpt  # saved in training process
- last.ckpt                # saved at the end of training process
- best.ctpt                # selected the best model by validation score in evaluation process
- average.ckpt             # output of checkpoint weight averaging (CPA) at evaluation process
```


## Test only single checkpoint
```bash
python src/test.py --ckpt-path PATH/TO/CKPT --save-dir PATH/TO/TREES/ --metrics OriginalParseval
```


## Parse raw document (sequence of EDUs)
TBU


## Performance and Checkpoints
These are results evaluated by OriginalParseval.
(scores are a bit different from the paper due to model retraining.)

|Model       |LM     |CPA|Span |Nuc. |Rel. |Ful. |ckpt|
|------------|-------|:-:|----:|----:|----:|----:|----|
|Shift-Reduce|DeBERTa|   | 77.0| 67.2| 57.5| 55.5| URL|
|Shift-Reduce|DeBERTa| x | 78.3| 69.0| 58.2| 56.2| URL|


## Reference
```text
TBU
```


## LICENSE

This software is released under the NTT License, see `LICENSE.txt`.

According to the license, it is not allowed to create pull requests. Please feel free to send issues.
