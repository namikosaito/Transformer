from os.path import join
from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        critetion: nn.Module,
        device: torch.device,
    ) -> None:
        self.net = net
        self.optimizer = optimizer
        self.critetion = critetion
        self.device = device
        self.net = self.net.to(self.device)

    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.critetion(preds, labels)

    def train_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        self.net.train()
        output = self.net(src, tgt)

        tgt = tgt[:, 1:]  # decoderからの出力は1 ~ max_lenまでなので、0以降のデータで誤差関数を計算する
        output = output[:, :-1, :]  #

        # calculate loss
        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )

        # calculate bleu score
        _, output_ids = torch.max(output, dim=-1)
        bleu_score = self.bleu_score(tgt, output_ids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, output, bleu_score

    def val_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        self.net.eval()
        output = self.net(src, tgt)

        tgt = tgt[:, 1:]
        output = output[:, :-1, :]  #

        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )
        _, output_ids = torch.max(output, dim=-1)
        bleu_score = self.bleu_score(tgt, output_ids)

        return loss, output, bleu_score

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, print_log: bool = True
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        # train
        train_losses: List[float] = []
        train_bleu_scores: List[float] = []
        if print_log:
            print(f"{'-'*20 + 'Train' + '-'*20} \n")
        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _, bleu_score = self.train_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(
                    f"train loss: {loss.item()}, bleu score: {bleu_score},"
                    + f"iter: {i+1}/{len(train_loader)} \n"
                )

            train_losses.append(loss.item())
            train_bleu_scores.append(bleu_score)

        # validation
        val_losses: List[float] = []
        val_bleu_scores: List[float] = []
        if print_log:
            print(f"{'-'*20 + 'Validation' + '-'*20} \n")
        for i, (src, tgt) in enumerate(val_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _, bleu_score = self.val_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(f"train loss: {loss.item()}, iter: {i+1}/{len(val_loader)} \n")

            val_losses.append(loss.item())
            val_bleu_scores.append(bleu_score)

        return train_losses, train_bleu_scores, val_losses, val_bleu_scores

    def test(self, test_data_loader: DataLoader) -> Tuple[List[float], List[float]]:
        test_losses: List[float] = []
        test_bleu_scores: List[float] = []
        for i, (src, tgt) in enumerate(test_data_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _ = self.val_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            test_losses.append(loss.item())

        return test_losses, test_bleu_scores