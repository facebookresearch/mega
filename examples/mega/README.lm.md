# Language Modeling

## Data
We download raw data of wikitext-103 and enwik8 using the script from the [transformer-xl](https://github.com/kimiyoung/transformer-xl/blob/master/getdata.sh) repository.

Next preprocess/binarize the data (following is an example for wikitext-103):
```bash
TEXT=wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```
Our processed binarized data can be downloaded from [wt103](https://dl.fbaipublicfiles.com/mega/data/wt103_data_bin.zip) and [enwik8](https://dl.fbaipublicfiles.com/mega/data/enwik8_data_bin.zip)

## Model checkpoints
Checkpoints for [wikitext-103](https://dl.fbaipublicfiles.com/mega/wt103.zip) and [enwik8](https://dl.fbaipublicfiles.com/mega/enwik8.zip), which contain
the model weight, training script and the log file.

## Training

```bash
# Set up training envs. Same for all tasks.
seed=$SEED

DATA=</path/to/data-dir>
SAVE=</path/save/dir>
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh
```

```bash
# wikitext-103, use 24 40GB A100
srun --label python -u train.py ${DATA} \
    --seed ${seed} --ddp-backend no_c10d --max-target-positions 8096 --decoder-hidden-dim 2048 \
    --valid-subset valid --task language_modeling -a "mega_lm_adaptive_big" \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --decoder-n-dim 16 --decoder-chunk-size 1024 --normalize-before --no-affine-final-norm \
    --max-tokens 6144 --update-freq 1 \
    --variant-block-multiple-min 2 --variant-block-multiple-max 6 \
    --normalization-type 'layernorm' --truncation-length 8192 --rel-pos-bias "rotary" \
    --optimizer adam --lr 5e-3 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 0.25 \
    --lr-scheduler linear_decay --total-num-update 400000 --end-learning-rate 0.0 \
    --warmup-updates 24000 --warmup-init-lr '1e-07' \
    --criterion adaptive_loss \
    --dropout 0.3 --attention-dropout 0.1 --hidden-dropout 0.1 --weight-decay 0.1 \
    --max-update 400000 \
    --no-epoch-checkpoints \
    --sample-break-mode 'complete'\
    --valid-block "splits:10" \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0
```

```bash
# enwik8, use 8 40GB A100
python -u train.py ${DATA} \
    --seed ${seed} --ddp-backend no_c10d --max-target-positions 10384 \
    --valid-subset valid --task language_modeling -a "mega_lm_enwik8_base" \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --decoder-n-dim 16 --decoder-chunk-size 2048 --normalize-before --no-affine-final-norm \
    --max-tokens 8192 --update-freq 1 \
    --variant-block-multiple-min 2 --variant-block-multiple-max 4 \
    --normalization-type 'layernorm' --truncation-length 8192 --rel-pos-bias "rotary" \
    --optimizer adam --lr 5e-3 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 0.25 \
    --lr-scheduler linear_decay --total-num-update 400000 --end-learning-rate 0.0 \
    --warmup-updates 24000 --warmup-init-lr '1e-07' \
    --criterion 'cross_entropy' --share-decoder-input-output-embed \
    --dropout 0.1 --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.1 \
    --max-update 400000 \
    --no-epoch-checkpoints \
    --sample-break-mode 'complete'\
    --valid-block "splits:10" \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0
```

## Evaluation

```bash
# wikitext-103
python -u fairseq_cli/eval_mega_lm.py ${DATA} \
    --path {SAVE}/model.pt  --gen-subset test --max-tokens 5000000 --test-chunk-size 2048 --chunk-nums 2  --softmax-batch 1024 \
    --sample-break-mode 'complete' --valid-block "splits:10" --model-overrides '{"decoder_chunk_size": 2048,"max_tokens_valid": 5000000}'
```

```bash
# enwik8
# please read the base 2 loss as bpc
python -u fairseq_cli/eval_mega_lm.py ${DATA} \
    --path ${SAVE}/model.pt --gen-subset test --max-tokens 10000000 --test-chunk-size 4096 --chunk-nums 3 --softmax-batch 1024 \
    --sample-break-mode 'complete' --valid-block "splits:50" --model-overrides '{"decoder_chunk_size": 4096,"max_tokens_valid":1000000}'
```