#!/bin/bash - 

set -o nounset                              # Treat unset variables as an error
set -e

. ./path.sh

. ./cmd.sh # cmd import 

stage=0

train_nj=128
test_nj=4


help_message="$0 [options] #TODO"

. ./utils/parse_options.sh

#TODO use utils/require_argument_all.sh, utils/require_argument_any.sh, ... to check the parameters

if [ $stage -le 0 ] ; then
    echo "$0: Stage 0: Split data"

     #python maatool/cli/split_json.py \
     #   data/valid.jsonl \
     #   $(for i in $(seq $test_nj) ; do echo -n " data/split$test_nj/valid.$i.json" ; done)

     #python maatool/cli/split_json.py \
     #   data/test.jsonl \
     #   $(for i in $(seq $test_nj) ; do echo -n " data/split$test_nj/test.$i.json" ; done)

    python maatool/cli/split_json.py \
        data/train.jsonl \
        $(for i in $(seq $train_nj) ; do echo -n " data/split$train_nj/train.$i.json" ; done)

fi

if [ $stage -le 1 ] ; then 
    echo "$0: Stage 1: extract feats"
    #$cmd JOB=1:$test_nj data_feats/valid/log/feats.JOB.log \
    #    python maatool/cli/extract_feats_from_json.py \
    #        --uid_basename valid-JOB \
    #        data/split$test_nj/valid.JOB.json \
    #        ark,scp:data_feats/valid/feats.JOB.ark,data_feats/valid/feats.JOB.scp 
    [ ! -d data_feats/valid/ ] && mkdir -p data_feats/valid/
    python maatool/cli/extract_feats_from_json.py \
        data/valid.jsonl \
        ark,scp:data_feats/valid/feats.ark,data_feats/valid/feats.scp 



    #$cmd JOB=1:$test_nj data_feats/test/log/feats.JOB.log \
    #    python maatool/cli/extract_feats_from_json.py \
    #        --uid_basename test-JOB \
    #        data/split$test_nj/test.JOB.json \
    #        ark,scp:data_feats/test/feats.JOB.ark,data_feats/test/feats.JOB.scp 
    [ ! -d data_feats/test/ ] && mkdir -p data_feats/test/
    python maatool/cli/extract_feats_from_json.py \
        data/test.jsonl \
        ark,scp:data_feats/test/feats.ark,data_feats/test/feats.scp 



    $cmd JOB=1:$train_nj data_feats/train/log/feats.JOB.log \
        python maatool/cli/extract_feats_from_json.py \
            data/split$train_nj/train.JOB.json \
            ark,scp:data_feats/train/feats.JOB.ark,data_feats/train/feats.JOB.scp
fi


if [ $stage -le 2 ] ; then 
    echo "$0: Stage 2: Extract words from jsons"
    awk '{print "valid-"NR-1" "$1}' ./data/valid.ref > ./data_feats/valid/text

    # test doesn't have any words

    $cmd JOB=1:$train_nj data_feats/train/log/text.JOB.log \
        python maatool/cli/text_from_json.py \
        data/split$train_nj/train.JOB.json \
        data_feats/train/text.JOB
    cat data_feats/train/text.* | sort -n > data_feats/train/text
fi



if [ $stage -le 3 ] ; then 
    echo "$0 Stage 3: prepare additional data"

    for n in accepted suggestion_accepted ; do 
    #for n in suggestion_accepted ; do 
        python maatool/cli/split_json.py \
            data/$n \
            $(for i in $(seq $train_nj) ; do echo -n " data/split$train_nj/$n.$i.json" ; done)

        python maatool/cli/split_json.py \
            data/$n \
            $(for i in $(seq $train_nj) ; do echo -n " data/split$train_nj/$n.$i.json" ; done)

        $cmd JOB=1:$train_nj data_feats/$n/log/feats.JOB.log \
            python maatool/cli/extract_feats_from_json.py \
            data/split$train_nj/$n.JOB.json \
            ark,scp:data_feats/$n/feats.JOB.ark,data_feats/$n/feats.JOB.scp || touch data_feats/$n/.error

        $cmd JOB=1:$train_nj data_feats/$n/log/text.JOB.log \
            python maatool/cli/text_from_json.py \
            data/split$train_nj/$n.JOB.json \
            data_feats/$n/text.JOB

        cat data_feats/$n/text.* | sort -n > data_feats/$n/text
    done
fi

if [ $stage -le 4 ] ; then 
    echo "$0: Stage 4: Run sentencepiece train"
    mkdir -p data/tokenizer
    #cat  \
    #    ./data/voc.txt \
    #    ./data/valid.ref \
    #    <(awk '{print $2}' data_feats/train/text ) \
    #    <(awk '{print $2}' data_feats/accepted/text ) \
    #    <(awk '{print $2}' data_feats/suggestion_accepted/text ) > data/tokenizer/train_tokenizer
    #spm_train \
    #    --input=data/tokenizer/train_tokenizer \
    #    --model_prefix=data/tokenizer/bpe.500 \
    #    --vocab_size=500 \
    #    --character_coverage=1.0 \
    #    --model_type=bpe

    #paste ./data/voc.txt <(spm_encode --model=data/tokenizer/bpe.500 \
    #    --output_format=piece < ./data/voc.txt ) > data/tokenizer/voc.piece
fi



