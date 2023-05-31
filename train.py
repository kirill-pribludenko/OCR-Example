import sys
from itertools import groupby

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from colorama import Fore
from torchmetrics import CharErrorRate
from tqdm import tqdm

from dataset import CapchaDataset
from model import CRNN

gpu = torch.device("cuda")
epochs = 3
size_of_one_digit = 32
model_save_path = "./checkpoints"
cer = CharErrorRate()


def cer_metric(prediction, y_true, blank):
    prediction = prediction.to(torch.int32)
    prediction = prediction[prediction != blank].tolist()
    str_prediction = ''.join(map(str, prediction))

    y_true = y_true.to(torch.int32)
    y_true = y_true[y_true != train_ds.blank_label].tolist()
    str_y_true = ''.join(map(str, y_true))

    current_cer = cer(str_prediction, str_y_true).numpy()

    return current_cer


def train_one_epoch(model, criterion, optimizer, data_loader) -> None:
    model.train()
    train_cer = []
    for x_train, y_train in tqdm(
            data_loader,
            position=0,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
    ):
        batch_size = x_train.shape[0]
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        optimizer.zero_grad()
        y_pred = model(x_train.cuda())
        input_lengths = torch.IntTensor(batch_size).fill_(size_of_one_digit)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        _, max_index = torch.max(
            y_pred, dim=2
        )
        for i in range(batch_size):
            raw_prediction = list(
                max_index[:, i].detach().cpu().numpy()
            )
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label]
            )

            train_cer.append(cer_metric(prediction, y_train[i], train_ds.blank_label))

    train_mean_cer = round(np.array(train_cer).mean(), 4)
    print("TRAINING. Mean CER: ", train_mean_cer)


def evaluate(model, val_loader) -> float:
    model.eval()
    with torch.no_grad():
        val_cer = []
        for x_val, y_val in tqdm(
                val_loader,
                position=0,
                leave=True,
                file=sys.stdout,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
        ):
            batch_size = x_val.shape[0]
            x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
            y_pred = model(x_val.cuda())
            input_lengths = torch.IntTensor(batch_size).fill_(size_of_one_digit)
            target_lengths = torch.IntTensor([len(t) for t in y_val])
            criterion(y_pred, y_val, input_lengths, target_lengths)
            _, max_index = torch.max(y_pred, dim=2)
            for i in range(batch_size):
                raw_prediction = list(max_index[:, i].detach().cpu().numpy())
                prediction = torch.IntTensor(
                    [c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label]
                )
                val_cer.append(cer_metric(prediction, y_val[i], train_ds.blank_label))

        val_mean_cer = round(np.array(val_cer).mean(), 4)
        print("VALIDATION. Mean CER: ", val_mean_cer)

    return val_mean_cer


def test_model(model, test_ds, number_of_test_imgs: int = 10):
    model.eval()
    test_loader = data_utils.DataLoader(test_ds, batch_size=number_of_test_imgs)
    test_preds = []
    (x_test, y_test) = next(iter(test_loader))
    y_pred = model(
        x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).cuda()
    )
    _, max_index = torch.max(y_pred, dim=2)
    for i in range(x_test.shape[0]):
        raw_prediction = list(max_index[:, i].detach().cpu().numpy())
        prediction = torch.IntTensor(
            [c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label]
        )
        prediction = prediction.to(torch.int32)
        prediction = prediction[prediction != train_ds.blank_label].tolist()
        str_prediction = ' '.join(map(str, prediction))
        test_preds.append(str_prediction)

    for j in range(len(x_test)):
        mpl.rcParams["font.size"] = 8
        plt.imshow(x_test[j], cmap="gray")
        mpl.rcParams["font.size"] = 18
        str_y_true = y_test[j].to(torch.int32)
        str_y_true = str_y_true[str_y_true != train_ds.blank_label].tolist()
        str_y_true = ' '.join(map(str, str_y_true))
        plt.gcf().text(x=0.1, y=0.1, s="     Actual: " + str_y_true)
        plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + test_preds[j])
        plt.savefig(f"./output/plot_{j}.png")
        plt.clf()


if __name__ == "__main__":
    train_ds = CapchaDataset((4, 5))
    test_ds = CapchaDataset((4, 5), samples=100)
    train_loader = data_utils.DataLoader(train_ds, batch_size=64)
    val_loader = data_utils.DataLoader(test_ds, batch_size=1)

    model = CRNN(1, size_of_one_digit,
                 size_of_one_digit * 5, train_ds.num_classes).to(gpu)

    criterion = nn.CTCLoss(
        blank=train_ds.blank_label, reduction="mean", zero_infinity=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    current_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        train_one_epoch(model, criterion, optimizer, train_loader)
        acc = evaluate(model, val_loader)
        if acc > current_acc:
            model_out_name = model_save_path + f"/checkpoint_{epoch}.pt"
            torch.save(model.state_dict(), model_out_name)

    test_model(model, test_ds)
