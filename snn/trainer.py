import torch, torch.nn as nn
from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class TrainCfg:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    amp: bool = True


class SNNCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets)


class SNNTrainer:
    def __init__(self, model, criterion, cfg):
        self.model = model
        self.criterion = criterion
        self.cfg = cfg

    def fit(self, train_loader, val_loader=None, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp and device.type == "cuda")

        best_val = 0.0
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            total, correct, running_loss = 0, 0, 0.0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.cfg.amp and device.type == "cuda"):
                    logits, aux = self.model(xb)
                    loss = self.criterion(logits, yb)

                scaler.scale(loss).backward()

                if self.cfg.clip_grad:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                scaler.step(opt)
                scaler.update()

                running_loss += loss.item() * xb.size(0)
                total += xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()

            train_acc = correct / total
            train_loss = running_loss / total

            val_acc = None
            if val_loader is not None:
                val_acc, _ = self.evaluate(val_loader, device=device)
                best_val = max(best_val, val_acc)

            sched.step()
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f}"
                + (f" | val_acc={val_acc:.4f} (best {best_val:.4f})" if val_acc else "")
            )

    @torch.no_grad()
    def evaluate(self, loader, device=None):
        device = device or next(self.model.parameters()).device
        self.model.eval()

        total, correct = 0, 0
        aux = None

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, aux_tmp = self.model(xb)

            total += xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()

            if aux is None:
                aux = {
                    "spike_counts": {
                        k: s.sum(0).cpu() for k, s in aux_tmp["spike_counts"].items()
                    },
                    "membrane_traces": {
                        k: [v.cpu()] for k, v in aux_tmp["membrane_traces"].items()
                    },
                    "spike_traces": {
                        k: [s.cpu()] for k, s in aux_tmp["spike_traces"].items()
                    },
                }
            else:

                for k in aux_tmp["spike_counts"]:
                    aux["spike_counts"][k] += aux_tmp["spike_counts"][k].sum(0).cpu()

                    if k in self.model.cfg.record_layers:
                        if (aux["membrane_traces"][k][0].shape == aux_tmp["membrane_traces"][k].shape):
                            aux["membrane_traces"][k].append(aux_tmp["membrane_traces"][k].cpu())
                            aux["spike_traces"][k].append(aux_tmp["spike_traces"][k].cpu())

            
        for k in aux_tmp["membrane_traces"]:
            aux["membrane_traces"][k] = torch.stack(aux["membrane_traces"][k], dim=0)
            aux["spike_traces"][k] = torch.stack(aux["spike_traces"][k], dim=0)

        acc = correct / total
        return acc, aux
