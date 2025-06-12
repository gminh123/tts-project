import torch
import os
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from model.tacotron2 import Tacotron2
from utils import LJSpeechDataset, collate_fn
from config import *

def masked_loss(pred, target, lens):
    mask = torch.zeros_like(target)
    for i, l in enumerate(lens):
        mask[i, :l, :] = 1
    return (torch.abs(pred - target) * mask).mean()

def run_validation(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for texts, text_lens, mels, mel_lens in val_loader:
            texts, text_lens = texts.to(device), text_lens.to(device)
            mels, mel_lens = mels.to(device), mel_lens.to(device)
            mel_inp = mels[:, :-1, :]
            mel_target = mels[:, 1:, :]

            with autocast(device_type="cuda"):
                mel_out, mel_post, _, _ = model(texts, mel_inp, text_lens)
                loss1 = masked_loss(mel_out, mel_target, mel_lens - 1)
                loss2 = masked_loss(mel_post, mel_target, mel_lens - 1)
                loss = loss1 + loss2

            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    dataset = LJSpeechDataset(f"{data_path}/metadata.csv", limit=6000)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    model = Tacotron2(vocab_size=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda")

    # Resume from last checkpoint if exists
    last_ckpt_path = "checkpoints/tacotron2_last.pth"
    if os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"🔄 Resumed training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        print(f"🔁 Epoch {epoch + 1}/{epochs}")
        loader_bar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)

        for batch_idx, (texts, text_lens, mels, mel_lens) in loader_bar:
            texts, text_lens = texts.to(device), text_lens.to(device)
            mels, mel_lens = mels.to(device), mel_lens.to(device)
            mel_inp = mels[:, :-1, :]
            mel_target = mels[:, 1:, :]

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                mel_out, mel_post, _, _ = model(texts, mel_inp, text_lens)
                loss1 = masked_loss(mel_out, mel_target, mel_lens - 1)
                loss2 = masked_loss(mel_post, mel_target, mel_lens - 1)
                loss = loss1 + loss2

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loader_bar.set_description(f"Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = run_validation(model, val_loader, device)
        print(f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss
            }, "checkpoints/tacotron2_best.pth")
            print(f"💾 Saved best checkpoint at epoch {epoch+1}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss
            }, f"checkpoints/tacotron2_epoch{epoch+1}.pth")

        # Always save last checkpoint
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss
        }, "checkpoints/tacotron2_last.pth")