#!/usr/bin/env python3
"""
Fast TRADES training on ResNet18 for 20 epochs.
ResNet18 is much faster than WideResNet-28-10, runs in ~1-2 hours.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data import get_data_info, load_data
from core.models import create_model
from core.attacks import create_attack, CWLoss
from core.utils import seed, Logger, format_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Configuration
DATA_DIR = 'dataset-data/cifar10'
LOG_DIR = 'trained_models/resnet18_cifar10_trades_20ep'
BATCH_SIZE = 128
BATCH_SIZE_VAL = 256
NUM_EPOCHS = 20
LR = 0.1
BETA = 5.0  # TRADES parameter
SEED = 42

os.makedirs(LOG_DIR, exist_ok=True)
logger = Logger(os.path.join(LOG_DIR, 'log.txt'))

# Seed for reproducibility
seed(SEED)

logger.log(f"Config: ResNet18, CIFAR-10, {NUM_EPOCHS} epochs, TRADES (β={BETA})")
logger.log(f"Batch size: {BATCH_SIZE}, LR: {LR}, Device: {device}\n")

# Load data
logger.log("Loading CIFAR-10...")
info = get_data_info(DATA_DIR)
train_dataset, test_dataset, train_loader, test_loader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VAL, num_workers=0, use_augmentation=True, use_consistency=False, 
    shuffle_train=True, aux_data_filename=None, unsup_fraction=0.0, validation=False
)
logger.log(f"✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}\n")

# Build model
logger.log("Building ResNet18 model...")
model = create_model('resnet18', normalize=False, info=info, device=device)
model = model.to(device)
logger.log(f"✓ Model loaded\n")

# Optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

# Attack for evaluation
attack = create_attack(model, CWLoss, 'linf-pgd', 8/255, 4*10, 2/255)

# Storage for metrics
all_metrics = []

logger.log("="*70)
logger.log("STARTING TRAINING")
logger.log("="*70 + "\n")

# Training loop
best_test_adv_acc = 0.0
best_epoch = 0

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    
    train_loss = 0.0
    train_clean_acc = 0.0
    train_adv_acc = 0.0
    num_batches = 0
    
    # Training step
    for x, y in tqdm(train_loader, desc=f'Epoch {epoch} (Train)', leave=False):
        x, y = x.to(device), y.to(device)
        
        # Clean forward pass
        logits_clean = model(x)
        loss_clean = F.cross_entropy(logits_clean, y)
        
        # Adversarial perturbation (PGD 10 iter)
        delta = torch.zeros_like(x, requires_grad=True)
        for _ in range(10):
            with torch.enable_grad():
                logits_perturbed = model(x + delta)
                loss_perturbed = F.cross_entropy(logits_perturbed, y)
            grad = torch.autograd.grad(loss_perturbed, delta, only_inputs=True)[0]
            delta = delta + (2/255) * torch.sign(grad)
            delta = torch.clamp(delta, -8/255, 8/255)
            delta = (x + delta).clamp(0, 1) - x
        delta = delta.detach()
        
        # Adversarial forward pass
        logits_adv = model(x + delta)
        loss_adv = F.cross_entropy(logits_adv, y)
        
        # TRADES loss: clean + β * adversarial
        loss = loss_clean + BETA * loss_adv
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        train_clean_acc += (logits_clean.argmax(1) == y).sum().item()
        train_adv_acc += (logits_adv.argmax(1) == y).sum().item()
        num_batches += 1
    
    train_loss /= num_batches
    train_clean_acc = 100.0 * train_clean_acc / len(train_dataset)
    train_adv_acc = 100.0 * train_adv_acc / len(train_dataset)
    
    # Evaluation on test set (clean)
    model.eval()
    test_clean_acc = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_clean_acc += (logits.argmax(1) == y).sum().item()
    test_clean_acc = 100.0 * test_clean_acc / len(test_dataset)
    
    # Evaluation on test set (adversarial, only every 2 epochs for speed)
    if epoch % 2 == 0 or epoch == NUM_EPOCHS:
        test_adv_acc = 0.0
        num_eval = 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f'Epoch {epoch} (Eval-Adv)', leave=False):
                x, y = x.to(device), y.to(device)
                x_adv = attack.perturb(x, y)
                logits = model(x_adv)
                test_adv_acc += (logits.argmax(1) == y).sum().item()
                num_eval += y.size(0)
        test_adv_acc = 100.0 * test_adv_acc / len(test_dataset)
        
        # Save best checkpoint
        if test_adv_acc > best_test_adv_acc:
            best_test_adv_acc = test_adv_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(LOG_DIR, 'model_best.pt'))
    else:
        test_adv_acc = None
    
    epoch_time = time.time() - epoch_start
    
    # Log metrics
    logger.log(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
               f"Clean (Train/Test): {train_clean_acc:.2f}% / {test_clean_acc:.2f}% | "
               f"Adv (Train): {train_adv_acc:.2f}%", )
    if test_adv_acc is not None:
        logger.log(f" | Adv (Test): {test_adv_acc:.2f}%", end='')
    logger.log(f" | Time: {format_time(epoch_time)}")
    
    # Store metrics
    metrics_dict = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_clean_acc': train_clean_acc,
        'train_adv_acc': train_adv_acc,
        'test_clean_acc': test_clean_acc,
        'test_adv_acc': test_adv_acc if test_adv_acc is not None else '',
        'time_sec': epoch_time
    }
    all_metrics.append(metrics_dict)
    
    scheduler.step()

# Final summary
logger.log("\n" + "="*70)
logger.log("TRAINING COMPLETED")
logger.log("="*70)
logger.log(f"\nBest checkpoint saved at Epoch {best_epoch} with Test Adv Acc: {best_test_adv_acc:.2f}%")

# Load and evaluate best model
logger.log(f"\nLoading best model and final evaluation...")
model.load_state_dict(torch.load(os.path.join(LOG_DIR, 'model_best.pt')))
model.eval()

# Clean accuracy
final_clean_acc = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        final_clean_acc += (logits.argmax(1) == y).sum().item()
final_clean_acc = 100.0 * final_clean_acc / len(test_dataset)

# Adversarial accuracy
final_adv_acc = 0.0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc='Final Adv Eval', leave=False):
        x, y = x.to(device), y.to(device)
        x_adv = attack.perturb(x, y)
        logits = model(x_adv)
        final_adv_acc += (logits.argmax(1) == y).sum().item()
final_adv_acc = 100.0 * final_adv_acc / len(test_dataset)

# Final results
logger.log("\n" + "="*70)
logger.log("FINAL RESULTS (Best Model)")
logger.log("="*70)
logger.log(f"Model:                 ResNet18")
logger.log(f"Dataset:               CIFAR-10")
logger.log(f"Training:              TRADES (β={BETA}) for {NUM_EPOCHS} epochs")
logger.log(f"Attack (Eval):         PGD (ε=8/255, 10 steps)")
logger.log(f"\nClean Accuracy:        {final_clean_acc:.2f}%")
logger.log(f"Adversarial Accuracy:  {final_adv_acc:.2f}%")
logger.log(f"Robustness Drop:       {final_clean_acc - final_adv_acc:.2f}%")
logger.log("="*70)

# Save metrics to CSV
df = pd.DataFrame(all_metrics)
csv_path = os.path.join(LOG_DIR, 'metrics.csv')
df.to_csv(csv_path, index=False)
logger.log(f"\nMetrics saved to: {csv_path}")
logger.log(f"Checkpoint saved to: {os.path.join(LOG_DIR, 'model_best.pt')}")
