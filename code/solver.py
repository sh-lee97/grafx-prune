import pickle
import random
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
from data.load import load_metadata
from loss import MRSTFTLoss
from mixing_console import construct_mixing_console
from prune import prune_grafx, prune_parameters
from tqdm import tqdm
from utils import overlap_add

from grafx import processors
from grafx.data import NodeConfigs, convert_to_tensor
from grafx.draw import draw_grafx
from grafx.render import prepare_render, render_grafx, reorder_for_fast_render
from grafx.utils import count_nodes_per_type, create_empty_parameters


class MusicMixingConsoleSolver(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.__dict__.update(**args)
        self.save_hyperparameters()
        self.init_audio_processors()
        self.init_graph_and_parameters()
        if self.prune:
            self.init_pruning_data()
        self.match_loss = MRSTFTLoss()

        self.automatic_optimization = False

    def init_audio_processors(self):
        def load_single_audio_processor(processor_type):
            kwargs = {}
            match processor_type:
                case "compressor" | "noisegate" | "delay" | "reverb":
                    kwargs["flashfftconv"] = self.flashfftconv
                    kwargs["max_input_len"] = self.audio_len

            match processor_type:
                case "eq":
                    p = processors.ZeroPhaseFIREqualizer(**kwargs)
                case "gain_panning":
                    p = processors.StereoGain(**kwargs)
                case "stereo_imager":
                    p = processors.SideGainImager(**kwargs)
                case "compressor":
                    p = processors.ApproxCompressor(**kwargs)
                case "noisegate":
                    p = processors.ApproxNoiseGate(**kwargs)
                case "delay":
                    p = processors.StereoMultitapDelay(**kwargs)
                case "reverb":
                    p = processors.MidSideFilteredNoiseReverb(**kwargs)

            match processor_type:
                case "eq" | "reverb":
                    p = processors.GainStagingRegularization(p)

            p = processors.DryWet(p)
            return p

        audio_processors = {}
        for processor_type in set(self.processors):
            audio_processors[processor_type] = load_single_audio_processor(
                processor_type
            )

        self.audio_processors = nn.ModuleDict(audio_processors)

    def init_graph_and_parameters(self):
        self.node_configs = NodeConfigs(self.processors)

        metadata = load_metadata(self.dataset, self.song)
        G = construct_mixing_console(
            metadata=metadata,
            dry_insert_processors=self.processors,
            multi_insert_processors=self.processors,
            node_configs=self.node_configs,
        )

        render_order = ["in"] + self.processors + ["mix"] + self.processors + ["out"]
        render_order = [self.node_configs.node_type_to_index[t] for t in render_order]
        self.render_order = render_order

        self.G = reorder_for_fast_render(
            G,
            method="fixed",
            fixed_order=self.render_order,
        )
        fig, _ = draw_grafx(self.G, node_above="rendering_order")
        save_path = join(self.save_dir, f"{self.id}_full.pdf")
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        self.init_num_processors = count_nodes_per_type(self.G, self.processors)

        self.G_tensor = convert_to_tensor(self.G)

        render_data = prepare_render(self.G_tensor)
        self.render_data = self.transfer_batch_to_device(render_data, "cuda", 0)

        self.graph_parameters = create_empty_parameters(self.audio_processors, self.G)
        self.soft_mask = nn.Parameter(torch.zeros(self.G_tensor.num_nodes))

        print(self.graph_parameters)

    def init_pruning_data(self):
        self.current_ratio = {k: 1 for k in self.processors}
        self.min_loss = np.inf
        self.sparsity_coeff = 0.0

    @torch.no_grad()
    def update_graph_and_parameters(self, prune_mask):
        graph_parameters = prune_parameters(
            self.G_tensor, self.graph_parameters, prune_mask, self.node_configs
        )
        self.graph_parameters = graph_parameters
        print(self.graph_parameters)
        self.soft_mask = nn.Parameter(self.soft_mask[~prune_mask])
        G = prune_grafx(self.G, prune_mask)

        self.G = reorder_for_fast_render(
            G,
            method="fixed",
            fixed_order=self.render_order,
        )
        self.G_tensor = convert_to_tensor(self.G)
        render_data = prepare_render(self.G_tensor)
        self.render_data = self.transfer_batch_to_device(render_data, "cuda", 0)

        self.update_ratio()

    def update_ratio(self):
        T = self.G_tensor.node_types
        for node_type in self.processors:
            T_i = self.node_configs.node_type_to_index[node_type]
            mask_T = T == T_i
            n_total_T = self.init_num_processors[node_type]
            ratio_T = mask_T.long().sum() / n_total_T
            self.current_ratio[node_type] = ratio_T.item()

    def pre_load_eval_data(self):
        # As we repeatedly call the same source tensor for the pruning evaluation,
        # we simply pre-load all of them on GPU for speed.
        val_loader = self.trainer.datamodule.val_dataloader()
        batches = []
        for _, batch in enumerate(tqdm(val_loader, desc="pre-load eval data")):
            batch = self.transfer_batch_to_device(batch, "cuda", 0)
            batches.append(batch)
        self.eval_batches = batches

    def on_train_epoch_start(self):
        self.training = False
        if self.prune:
            if self.debug:
                self.pre_load_eval_data()
                self.prune_graph()
            else:
                if self.current_epoch == self.prune_start_epoch:
                    self.pre_load_eval_data()
                if self.current_epoch >= self.prune_start_epoch:
                    self.prune_graph()

        self.training = True
        # NOTE - this will way of configuration the optimizer will initialize the states
        # (e.g., momentums) every epoch. While updating the optimizer is necessary after the pruning,
        # We could have also passed the states from the previous optimizer.
        self.trainer.optimizers = self.configure_optimizers()

    @torch.no_grad()
    def prune_graph(self):
        print(f"current ratio: {self.current_ratio}")

        match self.prune_policy:
            case "weight":
                prune_mask = self.prune_graph_weight()
            case "brute_force":
                prune_mask = self.prune_graph_brute_force()
            case "hybrid":
                if self.current_epoch % 4 == 3:
                    prune_mask = self.prune_graph_brute_force()
                else:
                    prune_mask = self.prune_graph_weight()
            case _:
                assert False

        self.update_graph_and_parameters(prune_mask)
        print(f"updated ratio: {self.current_ratio}")

        weight = self.get_weight()
        current_loss = self.evaluate(weight, "pruned run")
        print(f"\nafter: {current_loss:.3f}")

        if self.save_intermediate_graphs:
            pickle_dir = join(self.save_dir, f"{self.id}_{self.current_epoch}.pickle")
            pickle.dump(self.G, open(pickle_dir, "wb"))
            fig, _ = draw_grafx(self.G, node_above="rendering_order")
            save_path = join(self.save_dir, f"{self.id}_{self.current_epoch}.pdf")
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

    def forward(self, batch, mask_weight):
        mix_pred, intermediates_list, _ = render_grafx(
            self.audio_processors,
            batch["source"].float(),
            self.graph_parameters,
            self.render_data,
            common_parameters={"drywet_weight": mask_weight},
            parameters_grad=self.training,
        )
        return mix_pred, intermediates_list

    def get_weight(self, prune_mask=None):
        weight = torch.sigmoid(self.soft_mask)
        if prune_mask is not None:
            weight = weight * ~prune_mask
        return weight

    def training_step(self, batch, idx):
        # forward
        mix = batch["mix"].float()
        weight = self.get_weight()
        mix_pred, intermediates_list = self.forward(batch, weight)
        reg_loss_dict = self.aggregate_processor_loss(intermediates_list)

        # match
        match_loss_dict = self.match_loss(mix, mix_pred)

        if self.prune:
            sparsity_loss = self.compute_sparsity_loss()
            reg_loss_dict["sparsity"] = sparsity_loss

        total_loss = match_loss_dict["match/full"] + sum(reg_loss_dict.values())

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

        # log
        if idx % self.num_steps_per_log == 0:
            weight_dict = self.log_weight()
            self.log_dict(**match_loss_dict, **reg_loss_dict, **weight_dict)

        return total_loss

    def aggregate_processor_loss(self, intermediates_list):
        loss = {}
        for intermediates in intermediates_list:
            for k, v in intermediates.items():
                if k in loss:
                    loss[k] = loss[k] + v
                else:
                    loss[k] = v
        if "gain_reg" in loss:
            loss["gain_reg"] = loss["gain_reg"] * self.gain_reg_weight
        if "radii_reg" in loss:
            loss["radii_reg"] = loss["radii_reg"] / 2
        return loss

    def compute_sparsity_loss(self):
        if self.current_epoch < self.prune_start_epoch:
            return 0.0
        if self.current_epoch < self.prune_start_epoch + self.sparsity_raise_epoch:
            self.sparsity_coeff += 1 / (
                self.sparsity_raise_epoch * self.steps_per_epoch
            )
        mask_weight = self.get_weight()
        sparsity_loss = torch.sum(mask_weight)
        sparsity_loss = sparsity_loss * self.sparsity_loss_weight * self.sparsity_coeff
        return sparsity_loss

    @torch.no_grad()
    def evaluate(self, weight, desc):
        losses = []
        for _, batch in enumerate(tqdm(self.eval_batches, desc=desc)):
            mix = batch["mix"].float()
            mix_pred, _ = self.forward(batch, weight)
            match_loss_dict = self.match_loss(mix, mix_pred)
            losses.append(match_loss_dict["match/full"].item())
        return np.mean(losses)

    @torch.no_grad()
    def prune_graph_weight(self):
        prune_mask = torch.zeros(
            self.G_tensor.num_nodes, dtype=torch.bool, device="cuda"
        )
        # current run
        current_loss = self.evaluate(self.get_weight(), "current run")
        print(f"current loss: {current_loss:.3f}")

        self.min_loss = min(self.min_loss, current_loss)
        print(f"min loss: {self.min_loss:.3f}")
        processors_to_search = self.processors.copy()
        temp_ratio = {k: 0 for k in self.processors}
        deltas = {k: 0.1 for k in self.processors}
        while True:
            if len(processors_to_search) == 0:
                break
            node_type = random.choice(processors_to_search)

            idx = self.node_configs.node_type_to_index[node_type]
            num = (self.G_tensor.node_types == idx).long().sum().item()
            if num == 0:
                processors_to_search.remove(node_type)
                continue

            min_delta = 1 / num
            if deltas[node_type] > min_delta:
                temp_delta = deltas[node_type]
            else:
                temp_delta = min_delta
                processors_to_search.remove(node_type)
            temp_ratio[node_type] += temp_delta

            mask, weight = self.compute_prune_mask(temp_ratio)
            pruned_loss = self.evaluate(
                weight, f"search: {node_type}: {temp_ratio[node_type]:.3f}"
            )
            print("")
            print(f"pruned loss: {pruned_loss:.3f}")
            if pruned_loss > self.min_loss + self.tolerance:
                print(f"over the tolerance")
                temp_ratio[node_type] -= temp_delta
                deltas[node_type] /= 2
                if len(processors_to_search) == 0:
                    break
            else:
                print(f"succesful")
                if temp_ratio[node_type] >= 1:
                    temp_ratio[node_type] = 1
                    if node_type in processors_to_search:
                        processors_to_search.remove(node_type)
                self.min_loss = min(self.min_loss, pruned_loss)
            if self.debug:
                break
        prune_mask = self.compute_prune_mask(temp_ratio)[0]
        return prune_mask

    @torch.no_grad()
    def prune_graph_brute_force(self):
        # current run
        prune_mask = torch.zeros(
            self.G_tensor.num_nodes, dtype=torch.bool, device="cuda"
        )

        current_loss = self.evaluate(self.get_weight(), "current run")
        print(f"current loss: {current_loss:.3f}")
        self.min_loss = min(self.min_loss, current_loss)
        print(f"min loss: {self.min_loss:.3f}")

        T = self.G_tensor.node_types.cuda()
        not_pruned_yet = torch.where(~prune_mask)[0]
        for node_type in ["in", "out", "mix"]:
            util_mask = (
                T[not_pruned_yet] != self.node_configs.node_type_to_index[node_type]
            )
            not_pruned_yet = not_pruned_yet[util_mask]

        not_pruned_yet = not_pruned_yet.tolist()
        random.shuffle(not_pruned_yet)

        for i in range(len(not_pruned_yet)):
            idx = not_pruned_yet[i]
            temp_prune_mask = prune_mask.clone()
            temp_prune_mask[idx] = True
            weight = self.get_weight(temp_prune_mask)
            pruned_loss = self.evaluate(weight, f"{i}/{len(not_pruned_yet)}")
            print(f"pruned loss: {pruned_loss:.3f}")
            if pruned_loss > self.min_loss + self.tolerance:
                print(f"over the tolerance")
            else:
                print(f"succesful")
                prune_mask = temp_prune_mask
            self.min_loss = min(self.min_loss, pruned_loss)
            if self.debug:
                break
        return prune_mask

    @torch.no_grad()
    def compute_prune_mask(self, ratio_dict):
        weight = self.get_weight()

        sorted_weight, sorted_idx = torch.sort(weight)
        sorted_weight, sorted_idx = sorted_weight.tolist(), sorted_idx.tolist()

        pruned_weight = weight.clone()
        prune_mask = torch.zeros_like(pruned_weight, dtype=torch.bool)

        for node_type, ratio in ratio_dict.items():
            T_int = self.node_configs.node_type_to_index[node_type]
            mask = self.G_tensor.node_types == T_int
            num_processors = torch.sum(mask.long()).item()
            num_discards = int(round(num_processors * ratio))
            n = 0
            for i in sorted_idx:
                if n == num_discards:
                    break
                if self.G_tensor.node_types[i] == T_int:
                    pruned_weight[i] = 0.0
                    prune_mask[i] = True
                    n += 1
        return prune_mask, pruned_weight

    def on_test_epoch_start(self):
        self.mix_list, self.mix_pred_list = [], []
        self.full_losses, self.sum_losses, self.diff_losses = [], [], []

    @torch.no_grad()
    def test_step(self, batch, idx):
        mix = batch["mix"].float()
        mix_pred, _ = self.forward(batch, self.get_weight())
        match_loss_dict = self.match_loss(mix, mix_pred)
        self.log_dict(**match_loss_dict)
        self.mix_pred_list.append(mix_pred.detach().cpu())
        self.mix_list.append(mix.detach().cpu())
        return match_loss_dict["match/full"]

    @torch.no_grad()
    def on_test_epoch_end(self):
        if self.debug:
            return
        mix = overlap_add(self.mix_list, sr=30000, eval_warmup_sec=1, crossfade_ms=2)
        mix_pred = overlap_add(
            self.mix_pred_list, sr=30000, eval_warmup_sec=1, crossfade_ms=2
        )
        sf.write(join(self.save_dir, "orig.wav"), mix[0].T.numpy(), self.sr)
        sf.write(
            join(self.save_dir, f"{self.id}_pred.wav"), mix_pred[0].T.numpy(), self.sr
        )

        if self.prune:
            fig, _ = draw_grafx(self.G, node_above="rendering_order")
            save_path = join(self.save_dir, f"{self.id}_pruned.pdf")
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

            self.transfer_batch_to_device(self.render_data, "cpu", 0)
            data = {
                "parameters": self.graph_parameters.cpu(),
                "weight": self.get_weight().cpu(),
                "G": self.G,
                "G_tensor": self.G_tensor,
                "ratio": self.current_ratio,
            }
            pickle.dump(
                data, open(join(self.save_dir, f"{self.id}_result.pickle"), "wb")
            )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=0, eps=1e-8
        )
        return [opt]

    def log_dict(self, **kwargs):
        super().log_dict(
            kwargs, prog_bar=True, logger=True, on_epoch=True, batch_size=1
        )

    def log_weight(self):
        weight = self.get_weight()
        T = self.G_tensor.node_types
        weight_dict = {}
        nums = []
        for node_type in self.processors:
            T_i = self.node_configs.node_type_to_index[node_type]
            mask_T = T == T_i
            n_total_T = self.init_num_processors[node_type]
            if torch.any(mask_T):
                weight_T = torch.sum(weight[mask_T]) / n_total_T
                weight_T = weight_T.item()
                weight_dict[f"weight/{node_type}"] = weight_T
            else:
                weight_dict[f"weight/{node_type}"] = 0.0
            if self.prune:
                num_T = mask_T.long().sum()
                ratio_T = num_T / n_total_T
                ratio_T = ratio_T.item()
                weight_dict[f"ratio/{node_type}"] = ratio_T
                nums.append(num_T.item())
        if self.prune:
            init_num_total = sum(self.init_num_processors.values())
            weight_dict["ratio/mean"] = sum(nums) / init_num_total
        return weight_dict
