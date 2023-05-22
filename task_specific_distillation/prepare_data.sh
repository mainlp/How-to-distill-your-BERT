for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json /content/drive/MyDrive/datasets/gpt2_bpe/encoder.json \
        --vocab-bpe /content/drive/MyDrive/datasets/gpt2_bpe/vocab.bpe \
        --inputs /content/drive/MyDrive/datasets/wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs /content/drive/MyDrive/datasets/wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 1; \
done

python preprocess.py \
    --only-source \
    --srcdict /content/drive/MyDrive/datasets/gpt2_bpe/dict.txt \
    --trainpref /content/drive/MyDrive/datasets/wikitext-103-raw/wiki.train.bpe \
    --validpref /content/drive/MyDrive/datasets/wikitext-103-raw/wiki.valid.bpe \
    --testpref /content/drive/MyDrive/datasets/wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 1
