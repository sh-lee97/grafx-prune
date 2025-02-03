import pickle
from functools import partial
from glob import glob
from os.path import basename, dirname, join, realpath

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
    num_inputs = len(glob(join(BASE_DIR, song, "*RAW", "*.wav")))
    if num_inputs < min_num_inputs:
        return False
    if num_inputs > max_num_inputs:
        return False
    song = basename(song)
    return True


def get_medley_song_list(
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
        case "valid_and_test":
            song_list = sorted(song_list[n_train:])
        case "all":
            song_list = sorted(song_list)
        case _:
            assert False

    assert len(song_list) != 0
    song_list = [basename(s) for s in song_list]

    return song_list


def load_medley_metadata(song="Maroon5_ThisLove"):
    metadata = {}
    metadata["song"] = song
    metadata["dataset"] = "medley"
    metadata["base"] = BASE_DIR

    song_dir = join(BASE_DIR, song)
    metadata["song_dir"] = song_dir

    raw_metadata_dir = join(song_dir, f"{song}_METADATA.yaml")
    raw_metadata = safe_load(open(raw_metadata_dir, "r"))

    correspondence_data = {}
    for v in raw_metadata["stems"].values():
        multi = v["filename"]
        dry_files = [x["filename"] for x in v["raw"].values()]
        correspondence_data[multi] = dry_files
    correspondence_data = dict(matched=correspondence_data)
    metadata["correspondence"] = correspondence_data

    matched_dry_dirs = flatten(list(correspondence_data["matched"].values()))
    dry_dir = join(song_dir, f"{song}_RAW")
    metadata["matched_dry_dirs"] = [join(dry_dir, d) for d in matched_dry_dirs]

    matched_multi_dirs = list(correspondence_data["matched"].keys())
    multi_dir = join(song_dir, f"{song}_STEMS")
    metadata["matched_multi_dirs"] = [join(multi_dir, d) for d in matched_multi_dirs]

    metadata["mix_dir"] = join(song_dir, f"{song}_MIX.wav")
    metadata["total_len"] = sf.info(metadata["mix_dir"]).frames

    return metadata
