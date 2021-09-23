#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --mode 'unroll' --use-reg 1 --perturbation-target 0 1 \
    --init-std 1e-4 --inner-lr 1e-4 --eps 0.3 --inner-steps 1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 9, "lenpen": 1.5}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --log-format json --log-interval 5 \
    --num-workers 10 \
    --update-freq 1 \
    --ddp-backend no_c10d \
    --seed 1 \
