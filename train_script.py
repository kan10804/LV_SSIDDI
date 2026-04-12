import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import optim
import torch.nn as nn
from sklearn import metrics

import models
from data_preprocessing import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS

# ================= SEED =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ================= PARAM =================
class Args:
    n_atom_feats = TOTAL_ATOM_FEATS
    n_atom_hid = 128
    rel_total = 86
    lr = 3e-4
    n_epochs = 100
    kge_dim = 64
    batch_size = 256
    weight_decay = 5e-4
    neg_samples = 2
    patience = 10

args = Args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# Load data
train_df = pd.read_csv('/content/drive/MyDrive/data_clean/twosides/ddi_training.csv')
val_df   = pd.read_csv('/content/drive/MyDrive/data_clean/twosides/ddi_validation.csv')
test_df  = pd.read_csv('/content/drive/MyDrive/data_clean/twosides/ddi_test.csv')

train_tup = list(zip(train_df.d1, train_df.d2, train_df.type))
val_tup   = list(zip(val_df.d1, val_df.d2, val_df.type))
test_tup  = list(zip(test_df.d1, test_df.d2, test_df.type))

train_data = DrugDataset(train_tup, neg_ent=args.neg_samples)
val_data   = DrugDataset(val_tup, disjoint_split=False)
test_data  = DrugDataset(test_tup, disjoint_split=False)

train_loader = DrugDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader   = DrugDataLoader(val_data, batch_size=args.batch_size * 2)
test_loader  = DrugDataLoader(test_data, batch_size=args.batch_size * 2)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# model and optimizer
model = models.SSI_DDI(
    args.n_atom_feats,
    args.n_atom_hid,
    args.kge_dim,
    args.rel_total,
    heads_out_feat_params=[32, 32, 32, 32],
    blocks_params=[2, 2, 2, 2]
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda epoch: 0.96 ** epoch
)

# balance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.3):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, p_score, n_score):
        p_prob = torch.sigmoid(p_score)
        n_prob = torch.sigmoid(n_score)

        p_loss = -self.alpha * (1 - p_prob) ** self.gamma * torch.log(p_prob + 1e-8)
        n_loss = -(1 - self.alpha) * (n_prob ** self.gamma) * torch.log(1 - n_prob + 1e-8)

        return p_loss.mean() + n_loss.mean()

loss_fn = FocalLoss()

# ================= UTIL =================
def compute_metrics(probas, labels):
    pred = (probas >= 0.5).astype(int)

    acc = metrics.accuracy_score(labels, pred)
    auc = metrics.roc_auc_score(labels, probas)
    f1  = metrics.f1_score(labels, pred)

    p, r, _ = metrics.precision_recall_curve(labels, probas)
    auprc = metrics.auc(r, p)

    return acc, auc, auprc, f1


def do_compute(batch):
    pos_tri, neg_tri = batch

    pos_tri = [t.to(device) for t in pos_tri]
    neg_tri = [t.to(device) for t in neg_tri]

    p_score = model(pos_tri)
    n_score = model(neg_tri)

    probas = torch.cat([
        torch.sigmoid(p_score),
        torch.sigmoid(n_score)
    ]).detach().cpu().numpy()

    labels = np.concatenate([
        np.ones(len(p_score)),
        np.zeros(len(n_score))
    ])

    return p_score, n_score, probas, labels

# train state
best_auc = 0
counter = 0

def train():
    global best_auc, counter

    print("Start training:", datetime.now())

    for epoch in range(1, args.n_epochs + 1):
        start = time.time()
        model.train()

        train_loss = 0
        all_probas, all_labels = [], []

        for batch in train_loader:
            p_score, n_score, probas, labels = do_compute(batch)

            loss = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()

      
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            train_loss += loss.item()
            all_probas.append(probas)
            all_labels.append(labels)

        train_loss /= len(train_loader)

        train_metrics = compute_metrics(
            np.concatenate(all_probas),
            np.concatenate(all_labels)
        )

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        all_probas, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                p_score, n_score, probas, labels = do_compute(batch)
                loss = loss_fn(p_score, n_score)

                val_loss += loss.item()
                all_probas.append(probas)
                all_labels.append(labels)

        val_loss /= len(val_loader)

        val_metrics = compute_metrics(
            np.concatenate(all_probas),
            np.concatenate(all_labels)
        )

        val_auc = val_metrics[1]

        # save best model
        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "auc": val_auc
            }, "/content/best_model.pth")

            print("Save best model!")
        else:
            counter += 1

        #   early stopping if epochs=10 without improvement
        if counter >= args.patience:
            print("Early stopping!")
            break

        scheduler.step()

        print(f"\nEpoch {epoch} | Time {time.time()-start:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train AUC: {train_metrics[1]:.4f} | Val AUC: {val_metrics[1]:.4f}")
        print(f"Train AUPR: {train_metrics[2]:.4f} | Val AUPR: {val_metrics[2]:.4f}")
        print("-"*50)


def test():
    checkpoint = torch.load("/content/best_model.pth")

    model.load_state_dict(checkpoint["model"])
    print(f"Loaded best model (Epoch {checkpoint['epoch']}, AUC={checkpoint['auc']:.4f})")

    model.eval()

    all_probas, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            _, _, probas, labels = do_compute(batch)
            all_probas.append(probas)
            all_labels.append(labels)

    probas = np.concatenate(all_probas)
    labels = np.concatenate(all_labels)

    acc, auc, auprc, f1 = compute_metrics(probas, labels)

    print("\n===== TEST RESULTS =====")
    print(f"ACC   : {acc:.4f}")
    print(f"AUC   : {auc:.4f}")
    print(f"AUPRC : {auprc:.4f}")
    print(f"F1    : {f1:.4f}")
    
    df = pd.DataFrame({
        "label": labels,
        "prob": probas
    })

    df.to_csv("/content/drive/MyDrive/data_clean/twosides/test_predictions.csv", index=False)

    # save metrics
    metrics_df = pd.DataFrame([{
        "ACC": acc,
        "AUC": auc,
        "AUPRC": auprc,
        "F1": f1
    }])

    metrics_df.to_csv("/content/drive/MyDrive/data_clean/twosides/test_metrics.csv", index=False)

    print("Saved: test_predictions.csv & test_metrics.csv")
train()
test()
