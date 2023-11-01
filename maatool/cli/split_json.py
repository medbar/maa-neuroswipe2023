from keyboard_start.tools.viz import read_swipe_events, plot_keyboard, plot_swipe
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from collections import defaultdict
import click
from pathlib import Path


@click.command()
@click.argument('in_json')
@click.argument('out_files', nargs=-1)
def split_json(in_json, out_files):
    num_files = len(out_files)
    assert num_files > 0
    Path(out_files[0]).parent.mkdir(parents=True, exist_ok=True)
    files = [open(f, 'w') for f in out_files]
    with open(in_json) as f_in:
        for i, line in enumerate(tqdm(f_in)):
            curr_split = i % num_files
            files[curr_split].write(line)
    [f.close() for f in files]


if __name__ == "__main__":
    split_json()
