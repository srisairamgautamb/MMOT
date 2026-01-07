"""
Main training script for Neural MMOT Solver.

Usage:
    python train.py --config configs/m4_optimized.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import create_model
from training.loss import MMOTLoss
from training.trainer import MMOTTrainer
from data.loader import get_dataloaders


def load_config(config_path):
    """Load configuration from YAML file, merging with default if needed."""
    # Load base config
    base_config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with specified config if it exists and is different from default
    config_path = Path(config_path)
    if config_path.name != 'default.yaml' and config_path.exists():
        with open(config_path, 'r') as f:
            override_config = yaml.safe_load(f)
        
        # Deep merge
        def deep_merge(base, override):
            for key, value in override.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(config, override_config)
    
    return config


def main(args):
    print("=" * 70)
    print("NEURAL MMOT SOLVER - TRAINING")
    print("=" * 70)
    
    # Load configuration
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.train_dir:
        config['data']['train_dir'] = args.train_dir
    if args.val_dir:
        config['data']['val_dir'] = args.val_dir    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Set device
    device = config['hardware'].get('device', 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"\nDevice: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        train_dir=config['data']['train_dir'],
        val_dir=config['data']['val_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 4)
    )
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {config['training']['batch_size']}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config['model'])
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Size: {model.count_parameters() * 4 / 1024 / 1024:.2f} MB")
    
    # Create loss function
    grid = torch.linspace(
        config['grid']['S_min'],
        config['grid']['S_max'],
        config['grid']['M']
    )
    
    loss_fn = MMOTLoss(
        grid=grid,
        epsilon=config['loss']['epsilon'],
        lambda_distill=config['loss']['lambda_distill'],
        lambda_martingale=config['loss']['lambda_martingale'],
        lambda_marginal=config['loss']['lambda_marginal']
    ).to(device)  # Move grid buffer to MPS
    
    print("\nLoss function:")
    print(f"  λ_distill: {config['loss']['lambda_distill']}")
    print(f"  λ_martingale: {config['loss']['lambda_martingale']}")
    print(f"  λ_marginal: {config['loss']['lambda_marginal']}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=config['optimizer']['betas']
    )
    
    # Create learning rate scheduler
    scheduler = None
    if config['scheduler']['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min']
        )
    
    # Create trainer
    trainer_config = {
        'log_dir': config['logging']['log_dir'],
        'checkpoint_dir': config['checkpoint']['save_dir'],
        'log_freq': config['logging']['log_freq'],
        'save_freq': config['checkpoint']['save_freq'],
        'gradient_clip': config['training'].get('gradient_clip', 1.0)
    }
    
    trainer = MMOTTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trainer_config
    )
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        early_stopping_patience=config['early_stopping'].get('patience', None)
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest validation loss: {history['best_val_loss']:.6f}")
    print(f"Final training loss: {history['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.6f}")
    
    print(f"\nCheckpoints saved to: {config['checkpoint']['save_dir']}")
    print(f"TensorBoard logs: {config['logging']['log_dir']}")
    print("\nView training curves:")
    print(f"  tensorboard --logdir {config['logging']['log_dir']}")
    
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Neural MMOT Solver')
    parser.add_argument('--config', type=str, default='../configs/m4_optimized.yaml',
                        help='Path to config file')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
