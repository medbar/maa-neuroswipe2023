#!/bin/bash - 

set -o nounset                              # Treat unset variables as an error
set -e

. ./path.sh

. ./cmd.sh # cmd import 

stage=6

train_nj=128
test_nj=4


help_message="$0 [options] #TODO"

. ./utils/parse_options.sh


if [ $stage -le 0 ] ; then
    echo "$0: Stage 0: Split data"

     python maatool/cli/split_json.py \
        data/valid.jsonl \
        $(for i in $(seq $test_nj) ; do echo -n " data/split$test_nj/valid.$i.json" ; done)

     python maatool/cli/split_json.py \
        data/test.jsonl \
        $(for i in $(seq $test_nj) ; do echo -n " data/split$test_nj/test.$i.json" ; done)

    python maatool/cli/split_json.py \
        data/train.jsonl \
        $(for i in $(seq $train_nj) ; do echo -n " data/split$train_nj/train.$i.json" ; done)

fi

if [ $stage -le 1 ] ; then 
    echo "$0: Stage 1: extract feats"
    $cmd JOB=1:$test_nj data_feats/valid/log/feats.JOB.log \
        python maatool/cli/extract_feats_from_json.py \
            --uid_basename valid-JOB \
            data/split$test_nj/valid.JOB.json \
            ark,scp:data_feats/valid/feats.JOB.ark,data_feats/valid/feats.JOB.scp 
    [ ! -d data_feats/valid/ ] && mkdir -p data_feats/valid/
    python maatool/cli/extract_feats_from_json.py \
        data/valid.jsonl \
        ark,scp:data_feats/valid/feats.ark,data_feats/valid/feats.scp 



    $cmd JOB=1:$test_nj data_feats/test/log/feats.JOB.log \
        python maatool/cli/extract_feats_from_json.py \
            --uid_basename test-JOB \
            data/split$test_nj/test.JOB.json \
            ark,scp:data_feats/test/feats.JOB.ark,data_feats/test/feats.JOB.scp 
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

         ## I use additional data only for sentencepiece model
         #$cmd JOB=1:$train_nj data_feats/$n/log/feats.JOB.log \
         #    python maatool/cli/extract_feats_from_json.py \
         #    data/split$train_nj/$n.JOB.json \
         #    ark,scp:data_feats/$n/feats.JOB.ark,data_feats/$n/feats.JOB.scp || touch data_feats/$n/.error
 
         $cmd JOB=1:$train_nj data_feats/$n/log/text.JOB.log \
             python maatool/cli/text_from_json.py \
             data/split$train_nj/$n.JOB.json \
             data_feats/$n/text.JOB
 
         cat data_feats/$n/text.* | sort -n > data_feats/$n/text
     done
 fi


if [ $stage -le 4 ] ; then 
    echo "$0: Stage 4: Run sentencepiece train"
    exp_dir=exp/bpe500
    mkdir -p $exp_dir
    python maatool/cli/tokenize_text.py \
        --text_out_dir $exp_dir \
        --vocab_size 500 \
        $exp_dir/model  \
        ./data/voc.txt \
        ark:data_feats/valid/text \
        ark:data_feats/train/text \
        ark:data_feats/accepted/text \
        ark:data_feats/suggestion_accepted/text 
fi


if [ $stage -le 5 ] ; then 
    echo "$0: Stage 5: Get grid names"
    python maatool/cli/keyboard_from_json.py \
        data/valid.jsonl \
        data_feats/valid/grid_name

    python maatool/cli/keyboard_from_json.py \
        data/test.jsonl \
        data_feats/test/grid_name

     $cmd JOB=1:$train_nj data_feats/train/log/grid_name.JOB.log \
        python ./maatool/cli/keyboard_from_json.py \
        data/split$train_nj/train.JOB.json \
        data_feats/train/grid_name.JOB
    cat data_feats/train/grid_name.* | sort -n > data_feats/train/grid_name

    
fi


if [ $stage -le 6 ] ; then 
    echo "$0: Stage 6: Grid names for additional data"
    for n in accepted suggestion_accepted ; do 
        $cmd JOB=1:$train_nj data_feats/$n/log/grid_name.JOB.log \
            python maatool/cli/keyboard_from_json.py \
            data/split$train_nj/$n.JOB.json \
            data_feats/$n/grid_name.JOB

        cat data_feats/$n/grid_name.* | sort -n > data_feats/$n/grid_name
    done
fi


if [ $stage -le 7 ] ; then 
    echo "$0: Stage 7: Run train"
    python ./train_conformer_v1.py
fi

echo "$0: What's all. Load checkpoint in after_deadline.ipynb and make submit"
exit
