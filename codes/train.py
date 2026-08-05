import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np
from tqdm import tqdm
from math import exp, ceil
import wandb
import random
from google.colab import drive

# ==========================================
# VERSION CONTROL & CONFIG DICTIONARY
# ==========================================
WANDB_PROJECT = "behzadGPT"
WANDB_RUN_NAME = "1_base(9)"
CONFIG = {
    # Architecture
    "vocab_size": 8192,
    "d_model": 448,
    "d_ff": 1280,         # 2.6667x the d_model but closest 256 multiple
    "context_length": 512,
    "d_head": 64,
    "layer_count": 9,

    # Training Hyperparameters
    "batch_size": 32,
    "accumulate": 4,      # this simulates batch size of 128 on small GPU hehe
    "max_lr": 1e-3,
    "weight_decay": 0.1,
    "label_smoothing": 0.1,
    "min_lr_perc": 0.1,
    "warmup_perc": 0.04,
    "init_std": 0.02,
    "max_grad_norm": 1.0,

    # Data & System
    "load_train_data_path": "/content/drive/MyDrive/behzadGPT/dataset/train/train_merged.bin",
    "load_val_data_path": "/content/drive/MyDrive/behzadGPT/dataset/validation/validation_merged.bin",
    "train_data_path": "/content/train_merged.bin",
    "val_data_path": "/content/validation_merged.bin",
    "checkpoint_path": "/content/drive/MyDrive/behzadGPT/checkpoint/",
    "num_workers": 4,
    "seed": 42,
    "eval_interval": 500,
    "train_log_interval": 10,
    "continue_log": 8680
}

# ==========================================
# DATASET
# ==========================================
import torch
from torch.utils.data import Dataset
import numpy as np

class Data(Dataset):
    def __init__(self, path, seq_len):
        super().__init__()
        self.path = path
        self.seq_len = seq_len
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, index):
        index = int(index)
        start_idx = index * self.seq_len
        chunk = self.data[start_idx : start_idx + self.seq_len + 1].astype(np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# MODEL COMPONENTS
# ==========================================
def precompute_rope_freqs(d_head, seq_len):
    i = torch.arange(0, d_head // 2, dtype=torch.float32)
    theta = 10000.0 ** (-2 * i / d_head)
    m = torch.arange(0, seq_len, dtype=torch.float32)

    freqs = torch.outer(m, theta)
    freqs = torch.cat([freqs, freqs], dim=-1)

    return torch.sin(freqs), torch.cos(freqs)


def apply_RoPE(x, sin, cos):
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head)
    cos = cos.unsqueeze(0).unsqueeze(0)

    half_d = x.shape[-1] // 2
    rotated_x = torch.cat([-x[..., half_d:], x[..., :half_d]], dim=-1)
    return (x * cos) + (rotated_x * sin)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, context_length):
        super().__init__()
        if d_model % d_head != 0:
            raise ValueError("d_model and d_head are not divisible.")
        self.num_heads = d_model // d_head
        self.d_head = d_head
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, sin, cos):
        B, S, _ = x.shape

        Q = apply_RoPE(self.Q(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2), sin[:S], cos[:S])
        K = apply_RoPE(self.K(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2), sin[:S], cos[:S])
        V = self.V(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        logits = (Q @ K.transpose(-1, -2)) / (self.d_head ** 0.5)

        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(logits, dim=-1)
        heads_out = attn_weights @ V

        heads_out = heads_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.O(heads_out)


class SwiGELU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, seq_len, d_ff):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, d_head, seq_len)
        self.ffn = SwiGELU_FFN(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attention.O.special_init=True
        self.ffn.down.special_init=True

    def forward(self, x, sin, cos):
        x = x + self.attention(self.norm1(x), sin, cos)
        return x + self.ffn(self.norm2(x))


class behzadGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        sin, cos = precompute_rope_freqs(config["d_head"], config["context_length"])
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

        self.layers = nn.ModuleList([
            TransformerBlock(
                config["d_model"],
                config["d_head"],
                config["context_length"],
                config["d_ff"]
            ) for _ in range(config["layer_count"])
        ])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, self.sin, self.cos)
        return torch.matmul(x, self.embeddings.weight.T)

def init_model(model, layers, std=0.02):
    if isinstance(model, nn.Linear):
        if hasattr(model, "special_init"):
            torch.nn.init.normal_(model.weight, mean = 0.0, std = std * (2 * layers)**-0.5)
        else:
            torch.nn.init.normal_(model.weight, mean = 0.0, std = std)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)
    elif isinstance(model, nn.RMSNorm):
        torch.nn.init.ones_(model.weight)
    elif isinstance(model, nn.Embedding):
        torch.nn.init.normal_(model.weight, mean = 0.0, std = std)

# ==========================================
# CHECKPOINT FUNCTIONS
# ==========================================
def save_checkpoint(global_counter, model, lr_scheduler, scaler, optimizer):
    checkpoint = {
        "global_counter": global_counter,
        "model_state_dict": model.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "wandb_run_id": wandb.run.id
    }

    os.makedirs(CONFIG["checkpoint_path"], exist_ok=True)
    temp_path = os.path.join(CONFIG["checkpoint_path"], "temp.tmp")
    final_path = os.path.join(CONFIG["checkpoint_path"], "checkpoint.pth")

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, final_path)

def load_checkpoint(model, lr_scheduler, scaler, optimizer, device):
    target_path = os.path.join(CONFIG["checkpoint_path"], "checkpoint.pth")
    print(f"Loading checkpoint from: {target_path}")
    checkpoint = torch.load(target_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["global_counter"], checkpoint["wandb_run_id"]

# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
if __name__=="__main__":

    drive.mount("/content/drive")
    if os.path.exists(CONFIG["train_data_path"]):
        print("Dataset already downloaded.")
    else:
        !cp "{CONFIG['load_train_data_path']}" /content/
        !cp "{CONFIG['load_val_data_path']}" /content/
        print("Dataset downloaded.")

    base_seed = CONFIG["seed"]
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)

    train_dataset = Data(CONFIG["train_data_path"], CONFIG["context_length"])
    val_dataset = Data(CONFIG["val_data_path"], CONFIG["context_length"])

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    model = behzadGPT(CONFIG)
    model.apply(lambda m: init_model(m, CONFIG["layer_count"], CONFIG["init_std"]))
    model.to(device)

    loss_func = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["max_lr"], weight_decay=CONFIG["weight_decay"])

    total_steps = ceil(len(train_dataset) / (CONFIG["batch_size"] * CONFIG["accumulate"]))
    warmup_steps = int(total_steps * CONFIG["warmup_perc"])

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - warmup_steps),
        eta_min=CONFIG["max_lr"] * CONFIG["min_lr_perc"]
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    amp_enabled = (device_type == "cuda")
    scaler = GradScaler(device=device_type, enabled=amp_enabled)

    global_counter = 0
    resume_status = None
    wandb_run_id = None

    checkpoint_file = os.path.join(CONFIG["checkpoint_path"], "checkpoint.pth")
    if os.path.exists(checkpoint_file):
        global_counter, wandb_run_id = load_checkpoint(model, lr_scheduler, scaler, optimizer, device)
        resume_status = "must"

    g = torch.Generator()
    g.manual_seed(base_seed)

    dataset_len = len(train_dataset)
    dtype = torch.int32 if dataset_len < 2_000_000_000 else torch.int64
    all_indices = torch.randperm(dataset_len, generator=g, dtype=dtype)

    samples_consumed = global_counter * CONFIG["batch_size"] * CONFIG["accumulate"]
    remaining_indices = all_indices[samples_consumed:]

    train_subset = torch.utils.data.Subset(train_dataset, remaining_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=seed_worker,
        persistent_workers=True if CONFIG["num_workers"] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"] * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=CONFIG["num_workers"],
        worker_init_fn=seed_worker
    )

    wandb.login()
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, resume=resume_status, id=wandb_run_id, config=CONFIG)

    train_loss = 0.0
    eval_interval = CONFIG["eval_interval"]
    batch_count = total_steps
    train_log_interval = CONFIG["train_log_interval"]
    train_loss_fast = 0.0

    train_bar = tqdm(train_loader, desc=f"Training")

    model.train()
    optimizer.zero_grad()
    for i, (x, y) in enumerate(train_bar):
        x, y = x.to(device), y.to(device)

        with autocast(device_type=device_type, enabled=amp_enabled):
            output = model(x)
            loss = loss_func(output.view(-1, output.size(-1)), y.view(-1)) / CONFIG["accumulate"]

        scaler.scale(loss).backward()
        train_loss += loss.item()
        train_loss_fast += loss.item()

        if (i + 1) % CONFIG["accumulate"] == 0 or (i + 1) == len(train_loader):
            global_counter += 1
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_bar.set_description(f"Step {global_counter}/{total_steps}")

            if global_counter % train_log_interval == 0 or global_counter == batch_count:
                if global_counter > CONFIG["continue_log"]:
                    wandb.log({
                        "train/loss_fast": train_loss_fast / train_log_interval,
                        "learning_rate": lr_scheduler.get_last_lr()[0]
                    })
                train_loss_fast=0.0

            if global_counter % eval_interval == 0 or global_counter == batch_count:
                save_checkpoint(global_counter, model, lr_scheduler, scaler, optimizer)

                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        with autocast(device_type=device_type, enabled=amp_enabled):
                            val_output = model(val_x)
                            v_loss = loss_func(val_output.view(-1, val_output.size(-1)), val_y.view(-1))
                        val_loss += v_loss.item()

                if global_counter == batch_count:
                    train_loss /= batch_count % eval_interval
                else:
                    train_loss /= eval_interval

                val_loss /= len(val_loader)

                try:
                    train_perplexity = exp(train_loss)
                    val_perplexity = exp(val_loss)
                except OverflowError:
                    train_perplexity = float("inf")
                    val_perplexity = float("inf")

                wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/perplexity": train_perplexity,
                    "val/perplexity": val_perplexity,
                    "step": global_counter
                })

                train_loss = 0.0
                model.train()

print("Training complete.")