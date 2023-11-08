import click
from tqdm.auto import tqdm
from pathlib import Path
from kaldiio import WriteHelper

import sentencepiece as spm



def read_text_file(fname):
    with open(fname) as f:
        return f.readlines(), None

def read_ark_text_file(ark_rspec):
    assert ark_rspec.startswith('ark:'), f"wrong ark text rspec {ark_rspec}"
    fname = ark_rspec.split(':', maxsplit=1)[1]
    with open(fname) as f:
        data = [(uid, line) for uid, line in map(lambda x: x.split(" ", maxsplit=1), f.readlines())]
    return [line for u, line in data], [u for u, _ in data]



def sentence_generator(text_fnames):
    for fname in tqdm(text_fnames):
        if fname.startswith('ark:'):
            lines, _ = read_ark_text_file(fname)
        else:
            lines, _ = read_text_file(fname)
        for line in lines:
            yield line



@click.command()
@click.option("--vocab_size", default=500)
@click.option("--model_type", default="bpe")
@click.option("--text_out_dir", default=None)
@click.argument("model_prefix")
@click.argument("text_fnames", nargs=-1)
def spm_train(model_prefix, text_fnames, vocab_size=500, model_type='bpe', text_out_dir=None):
    spm.SentencePieceTrainer.train(sentence_iterator=sentence_generator(text_fnames),
                                   model_prefix=model_prefix,
                                   model_type=model_type,
                                   pad_id=0,
                                   pad_piece='<blk>',
                                   unk_id=3,
                                   vocab_size=vocab_size,
                                   character_coverage=1.0)

    if text_out_dir is not None:
        Path(text_out_dir).mkdir(parents=True, exist_ok=True)
        sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
        for fname in tqdm(text_fnames):
            lines, uids = read_ark_text_file(fname) if fname.startswith('ark:') else read_text_file(fname)
            encoded = [sp.encode(line, out_type="immutable_proto") for line in lines ]
#            print(encoded)
            if fname.startswith('ark:'):
                fname = fname.split(':', maxsplit=1)[1]
            fname=Path(fname)
            out_stem=f"{fname.parent.stem}-{fname.stem}"
            with open(f"{text_out_dir}/{out_stem}.piece", "w") as f:
                if uids is None:
                    f.write(''.join([" ".join(e.piece for e in line.pieces) + '\n' for line in encoded]))
                else:
                    f.write(''.join([f"{u} " + " ".join(e.piece for e in line.pieces) + '\n' for u, line in zip(uids, encoded)]))
            with open(f"{text_out_dir}/{out_stem}.int", "w") as f:
                if uids is None:
                    f.write(''.join([" ".join(str(e.id) for e in line.pieces) + '\n' for line in encoded]))
                else:
                    f.write(''.join([f"{u} " + " ".join(str(e.id) for e in line.pieces) + '\n' for u, line in zip(uids, encoded)]))
    pass


if __name__=="__main__":
    spm_train()
