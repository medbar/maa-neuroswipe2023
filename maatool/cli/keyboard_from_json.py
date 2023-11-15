from keyboard_start.tools.viz import read_swipe_events
import click
from tqdm.auto import tqdm
from pathlib import Path
from kaldiio import WriteHelper


@click.command()
@click.option('--uid_basename', default=None)
@click.option('--limit', default=-1)
@click.argument('in_json')
@click.argument('out_fname')
def text_from_json(in_json, out_fname, uid_basename=None, limit=-1):
    if uid_basename is None:
        uid_basename = Path(in_json).stem
    with open(out_fname, 'w') as f:
        for i, (event, ref) in enumerate(tqdm(read_swipe_events(in_json, limit=limit), total=6000000)):
            grid_name = event['curve']['grid']['grid_name']
            f.write(f"{uid_basename}-{i} {grid_name}\n")


if __name__ == "__main__":
    text_from_json()
