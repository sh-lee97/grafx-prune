import os
import pickle
from functools import partial
from glob import glob
from os.path import basename, dirname, isfile, join, realpath

import numpy as np
import soundfile as sf
from utils import flatten
from yaml import safe_load

script_path = dirname(realpath(__file__))
base_config_dir = join(script_path, "configs.yaml")
configs = safe_load(open(base_config_dir, "rb"))

BASE_DIR = configs["base_dir"]
SKIPS = configs["skips"]


def check_song(song, min_num_inputs=0, max_num_inputs=150):
    if any([s in song for s in SKIPS]):
        return False
    if not isfile(join(song, "alignment.pickle")):
        return False
    if not isfile(join(song, "correspondence.yaml")):
        return False
    metadata = safe_load(open(join(song, "correspondence.yaml"), "r"))
    num_dry = len(flatten(list(metadata.values())))
    if num_dry < min_num_inputs:
        return False
    if num_dry > max_num_inputs:
        return False
    return True


def get_mixing_secrets_song_list(
    mode, seed=0, n_valid=24, n_test=24, min_num_inputs=0, max_num_inputs=150
):

    song_list = sorted(glob(join(BASE_DIR, "*")))

    filter_func = partial(
        check_song, min_num_inputs=min_num_inputs, max_num_inputs=max_num_inputs
    )
    song_list = list(filter(filter_func, song_list))

    num_songs = len(song_list)
    assert num_songs != 0
    n_train = num_songs - n_valid - n_test
    n_valid = n_train + n_valid

    rng = np.random.RandomState(seed)
    rng.shuffle(song_list)

    match mode:
        case "train":
            song_list = sorted(song_list[:n_train])
        case "valid":
            song_list = sorted(song_list[n_train:n_valid])
        case "valid_yaml":
            song_list = safe_load(open(join(script_path, "valid.yaml"), "r"))
        case "test":
            song_list = sorted(song_list[n_valid:])
        case "all":
            song_list = sorted(song_list)
        case _:
            assert False

    assert len(song_list) != 0
    song_list = [basename(s) for s in song_list]
    return song_list


def load_mixing_secrets_metadata(song, sr=30000):
    metadata = {}
    metadata["song"] = song
    metadata["dataset"] = "mixing_secrets"
    metadata["base"] = BASE_DIR

    song_dir = join(BASE_DIR, song)
    metadata["song_dir"] = song_dir

    raw_metadata_dir = join(song_dir, f"correspondence.yaml")
    raw_metadata = safe_load(open(raw_metadata_dir, "r"))
    correspondence_data = dict(matched=raw_metadata)
    metadata["correspondence"] = correspondence_data

    matched_dry_dirs = flatten(list(correspondence_data["matched"].values()))
    dry_dir = [f.path for f in os.scandir(song_dir) if f.is_dir()][0]

    metadata["matched_dry_dirs"] = [join(dry_dir, d) for d in matched_dry_dirs]

    metadata["mix_dir"] = join(song_dir, f"mix.wav")
    metadata["total_len"] = sf.info(metadata["mix_dir"]).frames

    alignment_dir = join(song_dir, f"alignment.pickle")
    alignment_data = pickle.load(open(alignment_dir, "rb"))
    offset = alignment_data["mix.wav"] - alignment_data["rough_mix.wav"]
    offset_sample = int(round(offset * sr))
    metadata["dry_alignment"] = -offset_sample

    return metadata
