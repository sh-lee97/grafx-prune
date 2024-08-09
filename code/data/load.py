import librosa
import numpy as np
import soundfile as sf
import torch
import torch.fft
from data.medley.load import get_medley_song_list, load_medley_metadata
from data.mixing_secrets.load import (
    get_mixing_secrets_song_list,
    load_mixing_secrets_metadata,
)


def detach_base_dir(x, d=4):
    return "/".join(x.split("/")[-d:])


def load_metadata(dataset, song):
    match dataset:
        case "medley":
            return load_medley_metadata(song)
        case "mixing_secrets":
            return load_mixing_secrets_metadata(song)
        case _:
            assert False


def get_song_list(dataset, mode, min_num_inputs=0, max_num_inputs=150):
    match dataset:
        case "medley":
            return get_medley_song_list(
                mode, min_num_inputs=min_num_inputs, max_num_inputs=max_num_inputs
            )
        case "mixing_secrets":
            return get_mixing_secrets_song_list(
                mode, min_num_inputs=min_num_inputs, max_num_inputs=max_num_inputs
            )
        case _:
            assert False


def load_track(wav_dir, start, audio_len, sr=30000, half_precision=True):
    """
    load a single wav track and return.
    start: read offset. if negative, left-pad zeros with that amount.
    """
    try:
        frames = sf.info(wav_dir).frames
    except:
        print("?", wav_dir)
        assert False

    read_start = start if start >= 0 else 0
    read_frames = audio_len if start >= 0 else audio_len + start
    if read_frames <= 0:
        wav = np.zeros((audio_len, 2))
    else:
        try:
            wav, wav_sr = sf.read(
                wav_dir,
                start=read_start,
                frames=read_frames,
                always_2d=True,
                fill_value=0.0,
            )
            if len(wav) != read_frames:
                print(wav.shape, read_frames, audio_len, start)
                wav = wav[:read_frames]
        except:
            print(wav_dir)
            assert False

        if start < 0:
            wav = np.pad(wav, ((-start, 0), (0, 0)))
        if wav.shape[-1] == 1:
            wav = np.concatenate([wav, wav], -1)  # FORCED STEREO <- is this right?
        if wav_sr != sr:
            wav = librosa.resample(
                wav.T, orig_sr=wav_sr, target_sr=sr, res_type="polyphase"
            ).T

    return wav


def load_song(
    metadata=None,
    audio_len=2**18 - 4410,
    sr=30000,
    load_source=True,
    load_unmatched_as_source=True,
    target_tracks=["mix"],
    omit_silent_tracks=False,
    start=None,
    as_tensor=True,
    half_precision=True,
):
    data = {}

    song = metadata["song"]
    song_dir = metadata["song_dir"]
    total_len = metadata["total_len"]
    dataset = metadata["dataset"]

    # sample audio region
    if start is None:
        start = np.random.randint(total_len - audio_len)
    if audio_len == -1:
        audio_len = total_len - start

    # load audio
    if load_source:
        source_track_dirs = metadata["matched_dry_dirs"].copy()
        if load_unmatched_as_source:
            source_track_dirs += metadata.get("unmatched_dirs", [])

        dry_align = metadata.get("dry_alignment", 0)
        multi_align = metadata.get("multi_alignment", 0)
        dry_start, multi_start = start - dry_align, start - multi_align

        source_tracks = []
        for track_dir in source_track_dirs:
            source_start = (
                dry_start
                if ("dry.wav" in track_dir.lower()) or dataset != "internal"
                else multi_start
            )
            track = load_track(track_dir, source_start, audio_len, sr, half_precision)
            source_tracks.append(track)
        if as_tensor:
            source_tracks = [
                torch.tensor(x, dtype=torch.half if half_precision else torch.float).T
                for x in source_tracks
            ]
            source_tracks = torch.stack(source_tracks)
        data["source"] = source_tracks
        metadata["source_dirs"] = source_track_dirs

    if "mix" in target_tracks:
        mix_track_dir = metadata["mix_dir"]

        mix_start = start
        mix = load_track(mix_track_dir, mix_start, audio_len, sr, half_precision)
        if as_tensor:
            mix = torch.tensor(
                mix, dtype=torch.half if half_precision else torch.float
            ).T
        mix = mix[None, :, :]
        data["mix"] = mix

    return data
