from keyboard_start.tools.viz import read_swipe_events, plot_keyboard, plot_swipe
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import click
from scipy.signal import resample
import math
from kaldiio import ReadHelper, WriteHelper
from pathlib import Path

from maatool.data.labels2id import LABEL2ID


def linear_coords(t, x, y, stride=3, max_t=3700):
    if t[0] != 0:
        print(f"Warning: {t} doesn't start this 0")
    curr_maxt = max(t)
    if curr_maxt > max_t:
        print(f"Warning: {curr_maxt=} > {max_t=}. Normalize t")
        t = np.asarray(t)//curr_maxt * max_t
    align = np.argsort(t)
    linear = [[t[align[0]], x[align[0]], y[align[0]]]]
    for i in align[1:]:  # , start=t[0]):
        dot = t[i]
        prev = linear[-1]
        diff = dot - prev[0]
        if diff == 0:
            # print(f"Warning: {diff=}. {i=} {t[max(0, i-5): i+5]=}")
            continue
        assert diff > 0, f"Something gone wrong. {diff}. {i=}, {t=}, {linear=}, {dot=}"
        x_step = (x[i] - prev[1]) / diff
        y_step = (y[i] - prev[2]) / diff
        for j in range(1, diff + 1):
            linear.append([prev[0] + j,
                           prev[1] + j * x_step,
                           prev[2] + j * y_step])

    linear = np.asarray(linear)
    num_samples, num_coords = linear.shape
    pad = (num_samples // stride + 1) * stride - num_samples
    if pad > 0:
        # print(pad, num_samples)
        linear = np.pad(linear, pad_width=((0, pad), (0, 0)), mode='edge')

    linear = linear.reshape(linear.shape[0] // stride, stride, num_coords).mean(axis=1)
    return linear  # (Time, [t, x, y]])


def get_feats_for_series(x, y, grid, label2id):
    x = np.asarray(x)
    y = np.asarray(y)
    assert len(x) == len(y)
    deltas = [None for _ in range(len(label2id))]
    # print(x, y)
    for button in grid['keys']:
        if 'label' not in button:
            continue
        if button['label'] not in label2id:
            print(button['label'])
            continue
        center_x = button['hitbox']['x'] + button['hitbox']['w'] / 2
        center_y = button['hitbox']['y'] + button['hitbox']['h'] / 2
        delta_x = (x - center_x) / grid['width']
        delta_y = (y - center_y) / grid['height']
        # print(delta_x, delta_y)

        delta_pow2 = delta_x ** 2 + delta_y ** 2
        deltas[label2id[button['label']]] = delta_pow2
    deltas = np.stack([np.ones(len(x)) if l is None else l for l in deltas]).T
    return np.sqrt(deltas)


def test_get_feats_for_series():
    x = [0, 0, 5, 10]
    y = [1, 40, 1, 50]
    label2id = {'<pad>': 0, 'A': 1, 'B': 2}
    grid = {"width": 10,
            "height": 100,
            "keys": [{"label": "A", "hitbox": {"x": 0, "y": 20, "w": 10, "h": 40}},
                     {"label": "B", "hitbox": {"x": 5, "y": 0, "w": 5, "h": 100}}]}
    feats = get_feats_for_series(x, y, grid, label2id)
    print(feats)
    assert (feats[:, 0] == 1).all()
    assert feats[1, 1] == 0.5
    assert np.isclose(feats[1, 2], 0.75663729752).all()


@click.command()
@click.option('--uid_basename', default=None)
@click.option('--stride', default=10) # 3700/10= 370 max batch
@click.option('--limit', default=-1)
@click.argument('in_json')
@click.argument('out_wspec')
def process_data(in_json, out_wspec, uid_basename=None, stride=8, limit=-1):
    print(f"Extract feats from {in_json} with {stride=}")
    if uid_basename is None:
        uid_basename = Path(in_json).stem
    with WriteHelper(out_wspec) as fout:
        for i, (event, ref) in enumerate(tqdm(read_swipe_events(in_json, limit=limit))):
            curve = event['curve']
            if not (len(curve['t']) == len(curve['x']) == len(curve['y'])):
                print(f"Warning: {i} element is broken. Skip it.")
                continue
            inear_txy = linear_coords(t=curve['t'], x=curve['x'], y=curve['y'], stride=stride)
            feats = get_feats_for_series(inear_txy[:, 1], inear_txy[:, 2], curve['grid'], label2id=LABEL2ID)
            fout(f"{uid_basename}-{i}", feats)


if __name__ == "__main__":
    process_data()


