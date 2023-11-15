
. ./path.sh


#cat ./data_feats/train/feats.7.scp \
#    ./data_feats/test/feats.scp | shuf > data_feats/train7_sv16.feats.scp

#copy-feats scp:data_feats/train7_sv16.feats.scp \
#    ark:data_feats/train7_sv16.feats.ark

#utils/filter_scp.pl ./data_feats/train/feats.7.scp exp/bpe500/train-text.int \
#    | cat - exp/v16.test.int > exp/v16.train_test.int

python ./train_conformer_v1.14.py
