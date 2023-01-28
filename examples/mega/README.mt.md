# Mega on Neural Machine Translation

## Data

### WMT'16 English to German

First download and preprocess the WMT16 English to German data by following instructions:

```bash
# Download and prepare the data
cd examples/mega/
# WMT'16 data:
bash prepare-wmt16en2de.sh
cd ../../

# Binarize the dataset
TEXT=examples/mega/wmt16ende
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16ende_bpe32k \
    --nwordssrc 33000 --nwordstgt 33000 \
    --joined-dictionary \
    --workers 8
```

## Model Training

Train a ```Mega-base``` model on WMT'16 English to German, using 8 x V100 GPUs with 32G memory.

```bash
DATA_PATH=data-bin/wmt16ende_bpe32k
MODEL_PATH=<path of model>

python -u train.py $DATA_PATH \
    --seed 1 --ddp-backend c10d \
    --valid-subset valid -s 'en' -t 'de' \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a 'mega_wmt_en_de' --encoder-layers 6 --decoder-layers 6 \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --encoder-n-dim 16 --encoder-chunk-size -1 \
    --normalization-type 'layernorm' --truncation-length 0 \
    --optimizer adam --lr 1e-3 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --label-smoothing 0.1 --max-tokens 8192 --max-sentences 1024 --share-all-embeddings \
    --dropout 0.15 --attention-dropout 0.1 --hidden-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.05 \
    --lr-scheduler linear_decay --total-num-update 500000 --end-learning-rate 0.0 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --keep-last-epochs 5 --keep-interval-updates 1 --update-freq 1 --save-interval-updates 5000 \
    --save-dir $MODEL_PATH --log-format simple --log-interval 100 --num-workers 0
```

## Evaluation
Evaluate a model on test data of WMT16 English to German:

```bash
# calculate tokenized BLEU
python -u fairseq_cli/generate.py $DATA_PATH --gen-subset test -s en -t de --path ${MODEL_PATH}/checkpoint.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --lenpen 0.5 > ${MODEL_PATH}/gen.out
bash scripts/compound_split_bleu.sh ${MODEL_PATH}/gen.out

# calculate SacreBLEU
bash scripts/sacrebleu.sh wmt14/full en de ${MODEL_PATH}/gen.out
```
