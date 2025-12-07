# preprocess_ecg_shards_from_csv.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

# Where your ECG and label CSV shards live
X_DIR = Path("/path/to/ecg_shards")     # ecg_part0.npy, ecg_part1.npy, ...
CSV_DIR = Path("/path/to/csv_labels")   # labels_part0.csv, labels_part1.csv, ...

# Shard lists (keep the same order for X and CSVs)
x_paths = sorted([
    X_DIR / "ecg_part0.npy",
    X_DIR / "ecg_part1.npy",
    X_DIR / "ecg_part2.npy",
    # ...
])

csv_paths = sorted([
    CSV_DIR / "labels_part0.csv",
    CSV_DIR / "labels_part1.csv",
    CSV_DIR / "labels_part2.csv",
    # ...
])

# Name of label column in each CSV
LABEL_COL = "label"  # change if needed

# Output directory for preprocessed artifacts
OUT_DIR = Path("/path/to/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) Build per-shard label arrays + global metadata
# ============================================================

records = []      # rows: (global_idx, label, file_idx, local_idx[, id])
global_idx = 0
label_paths = []  # paths to labels_partX.npy we will create

for file_idx, (xp, cp) in enumerate(zip(x_paths, csv_paths)):
    print(f"Processing shard {file_idx}: ECG={xp.name}, CSV={cp.name}")

    # Open ECG shard as memmap just to get length (safe for huge files)
    X_mem = np.load(xp, mmap_mode="r")
    Ni_x = X_mem.shape[0]

    # Load CSV labels
    df = pd.read_csv(cp)

    if LABEL_COL not in df.columns:
        raise ValueError(f"{cp} does not contain label column '{LABEL_COL}'")

    labels = df[LABEL_COL].to_numpy()
    Ni_y = len(labels)

    if Ni_x != Ni_y:
        raise ValueError(
            f"Length mismatch shard {file_idx}: {xp} has {Ni_x} samples, "
            f"{cp} has {Ni_y} labels"
        )

    # Save this shard's labels as .npy
    y_path = OUT_DIR / f"labels_part{file_idx}.npy"
    np.save(y_path, labels)
    label_paths.append(y_path)

    # Optional: if you have an ID column, grab it; else None
    id_values = df["id"].to_numpy() if "id" in df.columns else None

    for local_idx in range(Ni_x):
        if id_values is not None:
            records.append(
                (global_idx, labels[local_idx], file_idx, local_idx, id_values[local_idx])
            )
        else:
            records.append(
                (global_idx, labels[local_idx], file_idx, local_idx, None)
            )
        global_idx += 1

# Build global metadata DataFrame
meta = pd.DataFrame(
    records,
    columns=["global_idx", "label", "file_idx", "local_idx", "id"]
).set_index("global_idx")

meta_path = OUT_DIR / "meta.csv"
meta.to_csv(meta_path, index=True)
print(f"\nSaved global metadata to: {meta_path}")
print("Total samples:", len(meta))


# ============================================================
# 2) Stratified train/val/test split on global_idx
# ============================================================

all_indices = meta.index.to_numpy()
all_labels  = meta["label"].to_numpy()

# 70% train, 15% val, 15% test
train_idx, temp_idx = train_test_split(
    all_indices,
    test_size=0.30,
    stratify=all_labels,
    random_state=42,
)

temp_labels = meta.loc[temp_idx, "label"].to_numpy()

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42,
)

train_idx = np.sort(train_idx)
val_idx   = np.sort(val_idx)
test_idx  = np.sort(test_idx)

np.save(OUT_DIR / "train_idx.npy", train_idx)
np.save(OUT_DIR / "val_idx.npy",   val_idx)
np.save(OUT_DIR / "test_idx.npy",  test_idx)

print("\nSaved splits in", OUT_DIR)
print("  train_idx.npy  size:", len(train_idx))
print("  val_idx.npy    size:", len(val_idx))
print("  test_idx.npy   size:", len(test_idx))


# ============================================================
# 3) Save shard path lists for convenience
# ============================================================

with open(OUT_DIR / "x_paths.txt", "w") as f:
    for p in x_paths:
        f.write(str(p) + "\n")

with open(OUT_DIR / "y_paths.txt", "w") as f:
    for p in label_paths:
        f.write(str(p) + "\n")

print("\nDone.")

# ==================== DATASET FOR MULTIPLE SHARDS =================
# ecg_dataset.py

import bisect
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiNpyECGDataset(Dataset):
    """
    Logically concatenates multiple (N_i, ...) ECG shards and their label shards.

    Each X file: (N_i, T, C) or (N_i, T)
    Each y file: (N_i,) or (N_i, K), aligned with that X file.
    """

    def __init__(
        self,
        x_paths: List[Path],
        y_paths: List[Path],
        mmap: bool = True,
        transform=None,
        target_transform=None,
        dtype_x: Optional[torch.dtype] = torch.float32,
        dtype_y: Optional[torch.dtype] = None,  # e.g. torch.long for classification
    ):
        assert len(x_paths) == len(y_paths), "x_paths and y_paths must have same length"
        self.x_paths = [Path(p) for p in x_paths]
        self.y_paths = [Path(p) for p in y_paths]
        self.mmap = mmap
        self.transform = transform
        self.target_transform = target_transform
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y

        self.X_files = []
        self.y_files = []
        self.lengths = []

        load_mode = "r" if mmap else None

        for xp, yp in zip(self.x_paths, self.y_paths):
            X = np.load(xp, mmap_mode=load_mode)  # memmap for ECGs
            y = np.load(yp)                       # labels usually small enough

            if len(X) != len(y):
                raise ValueError(f"Length mismatch in {xp} and {yp}: {len(X)} vs {len(y)}")

            self.X_files.append(X)
            self.y_files.append(y)
            self.lengths.append(len(X))

        # cumulative lengths for mapping global idx -> file/local idx
        self.cum_lengths = np.cumsum(self.lengths)

        print("Loaded MultiNpyECGDataset:")
        for i, (xp, n) in enumerate(zip(self.x_paths, self.lengths)):
            print(f"  Shard {i}: {xp.name} -> {n} samples")
        print(f"Total samples: {self.__len__()}")

    def __len__(self):
        return int(self.cum_lengths[-1])

    def _get_file_and_local_idx(self, idx: int):
        """
        Map a global index -> (file_idx, local_idx)
        """
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range [0, {self.__len__()})")

        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cum_lengths[file_idx - 1]
        return file_idx, local_idx

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._get_file_and_local_idx(idx)

        x = self.X_files[file_idx][local_idx]
        y = self.y_files[file_idx][local_idx]

        # numpy -> torch
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        if self.dtype_x is not None:
            x = x.to(self.dtype_x)
        if self.dtype_y is not None:
            y = y.to(self.dtype_y)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
# ================= TRAINING SCRIPT WITH MULTIPLE SHARDS ==============
# train_ecg_model.py

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ecg_dataset import MultiNpyECGDataset


# ============================================================
# 1) Load shard paths & splits from preprocessing
# ============================================================

PREPROC_DIR = Path("/path/to/preprocessed")

with open(PREPROC_DIR / "x_paths.txt", "r") as f:
    x_paths = [Path(line.strip()) for line in f if line.strip()]

with open(PREPROC_DIR / "y_paths.txt", "r") as f:
    y_paths = [Path(line.strip()) for line in f if line.strip()]

train_idx = np.load(PREPROC_DIR / "train_idx.npy")
val_idx   = np.load(PREPROC_DIR / "val_idx.npy")
test_idx  = np.load(PREPROC_DIR / "test_idx.npy")

print("Train/val/test sizes:", len(train_idx), len(val_idx), len(test_idx))


# ============================================================
# 2) Create logical-concat dataset and subsets
# ============================================================

full_ds = MultiNpyECGDataset(
    x_paths=x_paths,
    y_paths=y_paths,
    mmap=True,
    dtype_x=torch.float32,
    dtype_y=torch.long,   # change to float32 for regression
)

train_ds = Subset(full_ds, train_idx)
val_ds   = Subset(full_ds, val_idx)
test_ds  = Subset(full_ds, test_idx)


# ============================================================
# 3) DataLoaders
# ============================================================

BATCH_SIZE   = 64
NUM_WORKERS  = 4
PIN_MEMORY   = True

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# ============================================================
# 4) Example tiny training loop
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example model (assumes ECG is (B, T, C) and you permute to (B, C, T))
model = torch.nn.Sequential(
    torch.nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, padding=3),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool1d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(32, 2),  # e.g., 2 classes
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        # If your ECG shards are (N, T, C), x will be (B, T, C)
        if x.ndim == 3:
            x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    print(f"Epoch {epoch+1}, train loss: {running_loss / len(train_ds):.4f}")
