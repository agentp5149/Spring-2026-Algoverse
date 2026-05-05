"""
Train a small weather surrogate on processed ERA5 data.

Usage:
    python train_weather_surrogate.py \
        --data ../data/era5_processed/era5_processed.pt \
        --epochs 100 \
        --batch-size 32 \
        --hidden 64 \
        --output ../models/weather_surrogate.pt
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class WeatherUNet(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2), nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 4), nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2), nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU())
        self.dec1 = nn.Conv2d(hidden * 2, in_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        if d3.shape != e2.shape:
            d3 = d3[:, :, :e2.shape[2], :e2.shape[3]]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        if d2.shape != e1.shape:
            d2 = d2[:, :, :e1.shape[2], :e1.shape[3]]
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return x + out


def train(data_path, epochs=100, batch_size=32, lr=1e-3, hidden=64,
          output_path="models/weather_surrogate.pt"):
    data = torch.load(data_path, weights_only=False)
    train_X = data['train_inputs']
    train_Y = data['train_targets']
    cal_X = data['cal_inputs']
    cal_Y = data['cal_targets']

    in_channels = train_X.shape[1]
    print(f"Training weather surrogate")
    print(f"  Channels: {in_channels}")
    print(f"  Grid: {train_X.shape[2]}x{train_X.shape[3]}")
    print(f"  Train samples: {len(train_X)}")
    print(f"  Cal samples: {len(cal_X)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = WeatherUNet(in_channels, hidden=hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(train_X, train_Y),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            pred = model(batch_X)
            loss = loss_fn(pred, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, len(cal_X), batch_size):
                    cx = cal_X[i:i+batch_size].to(device)
                    cy = cal_Y[i:i+batch_size].to(device)
                    val_losses.append(loss_fn(model(cx), cy).item())
                val_loss = sum(val_losses) / len(val_losses)

            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                torch.save({
                    'model_state': model.state_dict(),
                    'in_channels': in_channels,
                    'hidden': hidden,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch + 1,
                }, output_path)
                print(f"    Saved best model (val_loss={best_val_loss:.6f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {output_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output", default="models/weather_surrogate.pt")
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size, args.lr, args.hidden, args.output)
