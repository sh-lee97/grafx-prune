import os
from datetime import datetime
from os.path import dirname, join, realpath

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from data.datamodule import SingleTrackOverfitDataModule
from data.load import get_song_list
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from solver import MusicMixingConsoleSolver
from tqdm import tqdm

torch.set_float32_matmul_precision("medium")


def create_log_directory(args):
    if args.debug:
        args.name = "debug"
    else:
        song = args.song.replace("/", "_")
        args.name = f"{args.dataset}_{song}"
    args.save_dir = join(args.base_dir, args.name)
    os.makedirs(args.save_dir, exist_ok=True)


def setup_loggers(args):
    args.wandb = not args.debug and args.wandb
    if args.wandb:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run = wandb.init(
            project=args.project,
            dir=join(args.save_dir),
            name=f"{args.song} {args.id} ({now})",
            reinit=True,
        )
        logger = WandbLogger()
        logger.experiment.config.update(OmegaConf.to_container(args))
    else:
        run = None
        logger = CSVLogger(save_dir=args.save_dir)
    return run, logger


def run_train(args):
    torch.manual_seed(42)
    create_log_directory(args)
    run, logger = setup_loggers(args)

    max_steps = args.total_epochs * args.steps_per_epoch

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=args.base_dir,
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        fast_dev_run=args.debug,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        # detect_anomaly=True,
    )

    args_cont = OmegaConf.to_container(args)
    solver = MusicMixingConsoleSolver(args_cont)
    datamodule = SingleTrackOverfitDataModule(args_cont)

    if len(args.processors) != 0:
        trainer.fit(solver, datamodule)
    trainer.test(solver, datamodule)

    if args.wandb:
        run.finish()


def run_trains(args):
    song_list = []
    for dataset in args.datasets:
        l = get_song_list(mode=args.dataset_split, dataset=dataset)
        song_list += [(dataset, song) for song in l]

    rng = np.random.RandomState(0)
    rng.shuffle(song_list)

    num_songs = len(song_list)
    num_songs_per_split = int(np.ceil(num_songs / args.num_splits))
    i_from = args.split_id * num_songs_per_split
    i_to = min(num_songs, (args.split_id + 1) * num_songs_per_split)

    for i in tqdm(range(i_from, i_to)):
        dataset, song = song_list[i]
        args.dataset = dataset
        args.song = song
        print_str = f"| {i}/{len(song_list)}: {dataset} - {song} |"
        print("=" * len(print_str))
        print(print_str)
        print("=" * len(print_str))
        run_train(args)


def setup_args():
    script_path = dirname(realpath(__file__))
    base_config_dir = join(script_path, "configs/base.yaml")
    args = OmegaConf.load(base_config_dir)

    cli_args = OmegaConf.from_cli()
    if "config" in cli_args:
        config_name = cli_args.pop("config")
        config_dir = join(script_path, f"configs/{config_name}.yaml")
        config_args = OmegaConf.load(config_dir)
        args = OmegaConf.merge(args, config_args)
    args = OmegaConf.merge(args, cli_args)

    print("=" * 40)
    print(OmegaConf.to_yaml(args), end="")
    print("=" * 40)

    return args


def inference(args):
    torch.manual_seed(42)
    create_log_directory(args)
    args_cont = OmegaConf.to_container(args)
    solver = MusicMixingConsoleSolver(args_cont).cuda()
    datamodule = SingleTrackOverfitDataModule(args_cont)
    solver.run_inference(args.pickle_path, datamodule)


if __name__ == "__main__":
    args = setup_args()
    if args.inference:
        inference(args)
    else:
        if args.multiple_runs:
            run_trains(args)
        else:
            run_train(args)
