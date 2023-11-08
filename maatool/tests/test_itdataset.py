import torch
from maatool.data.feats_itdataset import FeatsIterableDataset


def test_itdataset():
    ds = FeatsIterableDataset(feats_rspecifiers=['scp:data_feats/valid/feats.scp'],
                              targets_rspecifier="ark:exp/bpe500/valid-text.int")
    for item in ds:
        #print(item['targets'], type(item['targets']), item['targets'].dtype)
        assert item['targets'] is not None

    items = []
    for i, item in enumerate(ds):
        items.append(item)
        if i >= 60:
            break
    batch = ds.collate(items)
    assert isinstance(batch['targets'], torch.Tensor)

    ds = FeatsIterableDataset(feats_rspecifiers=['scp:data_feats/valid/feats.scp'])
    for item in ds:
        assert item['targets'] is None

    items = []
    for i, item in enumerate(ds):
        items.append(item)
        if i >= 60:
            break
    batch = ds.collate(items)
    assert batch['targets'] is None


if __name__ == "__main__":
    test_itdataset()
