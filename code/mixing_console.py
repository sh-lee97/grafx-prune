from os.path import basename, join

import numpy as np

from grafx.data import GRAFX


def true_or_false(p):
    return np.random.choice([True, False], p=[p, 1 - p])


def get_name(dataset, wav_dir):
    match dataset:
        case "medley":
            return basename(wav_dir)[-9:-4]
        case "mixing_secrets":
            return basename(wav_dir)[-9:-4]


def detach_base_dir(x, dataset):
    match dataset:
        case "medley":
            crop = -1
        case "mixing_secrets":
            crop = -1
        case _:
            assert False
    return join(*x.split("/")[crop:])


def construct_mixing_console(
    metadata,
    dry_insert_processors,
    multi_insert_processors,
    node_configs=None,
    load_unmatched_as_source=True,
):
    G = GRAFX(config=node_configs, invalid_op="error")

    correspondence_data = metadata["correspondence"]
    matched_dry_track_dirs = metadata["matched_dry_dirs"]
    unmatched_dirs = metadata.get("unmatched_dirs", [])
    dataset = metadata["dataset"]

    # dry insert chains
    dry_outs = []
    for dry_track in matched_dry_track_dirs:
        name = get_name(dataset, dry_track)
        in_id = G.add("in", name=name)  # , wav_dir=dry_track)
        start_id, end_id = G.add_serial_chain(dry_insert_processors)
        G.connect(in_id, start_id)
        dry_outs.append(end_id)

    # mix insert chains
    dry_track_dirs = [detach_base_dir(d, dataset) for d in matched_dry_track_dirs]
    matched_correspondences = correspondence_data["matched"]
    multi_tracks = list(matched_correspondences.keys())
    multi_outs = []
    for multi_track in multi_tracks:
        dry_tracks = matched_correspondences[multi_track]
        no_match = True
        for dry_track in dry_tracks:
            if dry_track in dry_track_dirs:
                no_match = False
                break
        if no_match:
            continue

        name = get_name(dataset, multi_track)
        mix_id = G.add("mix")
        for dry_track in dry_tracks:
            if dry_track in dry_track_dirs:
                dry_out = dry_outs[dry_track_dirs.index(dry_track)]
                G.connect(dry_out, mix_id)
        start_id, end_id = G.add_serial_chain(multi_insert_processors)
        G.connect(mix_id, start_id)
        multi_outs.append(end_id)

    # unmatched:
    if load_unmatched_as_source:
        for unmatched in unmatched_dirs:
            name = get_name(dataset, unmatched)
            in_id = G.add("in", name=name)  # , wav_dir=unmatched)
            start_id, end_id = G.add_serial_chain(
                dry_insert_processors + ["mix"] + multi_insert_processors
            )
            G.connect(in_id, start_id)
            multi_outs.append(end_id)

    out_id = G.add("out")
    for end_id in multi_outs:
        G.connect(end_id, out_id)

    return G
