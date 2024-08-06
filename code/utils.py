import torch


def flatten(l):
    return [item for sublist in l for item in sublist]


def unbatch_audio_list(audio_list):  # , batch_first):
    new_list = []
    for chunk in audio_list:
        if chunk.ndim == 4:
            new_list += list(chunk)
        else:
            new_list += [chunk]
    return new_list


@torch.no_grad()
def overlap_add(audio_list, sr=30000, eval_warmup_sec=1, crossfade_ms=2):
    audio_list = unbatch_audio_list(audio_list)
    crossfade = int(crossfade_ms / 1000 * sr)
    eval_warmup = int(sr * eval_warmup_sec)
    num_chunks = len(audio_list)
    _, num_channels, chunk_len = audio_list[0].shape
    total_len = chunk_len + (chunk_len - eval_warmup) * (num_chunks - 1)
    full_audio = torch.zeros(1, num_channels, total_len)
    init_window, mid_window, last_window = get_overlap_add_windows(
        chunk_len, eval_warmup, crossfade
    )
    for i, chunk in enumerate(audio_list):
        if i == 0:
            full_audio[:, :, :chunk_len] = chunk * init_window[None, None, :]
        else:
            start = chunk_len + (chunk_len - eval_warmup) * (i - 1) - crossfade
            end = start + chunk_len - eval_warmup + crossfade
            window = last_window if i == num_chunks - 1 else mid_window
            full_audio[:, :, start:end] += (
                chunk[:, :, eval_warmup - crossfade :] * window
            )
    return full_audio


def get_overlap_add_windows(chunk_len, eval_warmup, crossfade):
    init_window = torch.cat(
        [torch.ones(chunk_len - crossfade), torch.linspace(1, 0, crossfade)]
    )
    mid_window = torch.cat(
        [
            torch.linspace(0, 1, crossfade),
            torch.ones(chunk_len - eval_warmup - crossfade),
            torch.linspace(1, 0, crossfade),
        ]
    )
    last_window = torch.cat(
        [torch.linspace(0, 1, crossfade), torch.ones(chunk_len - eval_warmup)]
    )
    return init_window, mid_window, last_window
