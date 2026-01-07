import os
import sys
import torch
import numpy as np
import time
from pathlib import Path

# Add project root
sys.path.append('/Volumes/Hippocampus/Antigravity/MMOT')

from neural.models.architecture import create_model
from neural.inference.pricer import NeuralPricer
from neural.data.generator import generate_dataset
import yaml

def log(msg):
    print(f"\n[SYSTEM CHECK] {msg}")

def final_system_check():
    print("="*60)
    print("NEURAL MMOT: FINAL END-TO-END SYSTEM VERIFICATION")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. VERIFY DATA GENERATION
    # ---------------------------------------------------------
    log("1. Testing Data Generation...")
    try:
        # Create a temp dir
        test_dir = Path("data/system_check")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
        test_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate 5 instances
        # Function signature: generate_dataset(num_instances, output_dir, start_idx=0)
        generate_dataset(
            num_instances=5,
            output_dir=test_dir,
            start_idx=0
        )
        
        files = list(test_dir.glob("*.npz"))
        # Filter out macOS hidden files
        files = [f for f in files if not f.name.startswith("._")]
        
        if len(files) != 5:
            raise Exception(f"Generated {len(files)} files, expected 5")
            
        # Verify content
        data = np.load(files[0])
        if 'marginals' not in data or 'u_star' not in data:
            raise Exception("Missing keys in generated data")
            
        print("   ✅ Data Generation: SUCCESS (5 samples created)")
        
    except Exception as e:
        print(f"   ❌ Data Generation FAILED: {e}")
        return

    # ---------------------------------------------------------
    # 2. VERIFY MODEL & CONFIG
    # ---------------------------------------------------------
    log("2. Testing Model Initialization...")
    try:
        with open('configs/production_training.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Merge default defaults
        with open('configs/default.yaml', 'r') as f:
            defaults = yaml.safe_load(f)
            
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        final_config = deep_update(defaults, config)
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = create_model(final_config['model'])
        model.to(device)
        
        # Check Normalization Buffers
        if not hasattr(model, 'u_mean'):
             raise Exception("Model missing normalization buffers!")
             
        print(f"   ✅ Model Initialization: SUCCESS ({final_config['model']['grid_size']} grid)")
        
    except Exception as e:
        print(f"   ❌ Model Init FAILED: {e}")
        return

    # ---------------------------------------------------------
    # 3. VERIFY TRAINING LOOP (1 Epoch Dry Run)
    # ---------------------------------------------------------
    log("3. Testing Training Loop (1 Epoch)...")
    try:
        from neural.training.trainer import MMOTTrainer
        from neural.training.loss import MMOTLoss
        
        # Setup Data Loaders
        from neural.data.loader import get_dataloaders
        train_loader, val_loader = get_dataloaders(
            train_dir=test_dir, 
            val_dir=test_dir,
            batch_size=2,
            num_workers=0
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create Grid for Loss
        M = final_config['model']['grid_size']
        S_min = final_config['grid']['S_min']
        S_max = final_config['grid']['S_max']
        grid = torch.linspace(S_min, S_max, M).to(device)

        loss_fn = MMOTLoss(grid=grid, **final_config['loss'])
        
        trainer = MMOTTrainer(
            model, loss_fn, optimizer, None, device, final_config['training']
        )
        
        # Train 1 epoch
        # Ensure validation runs
        initial_loss, _ = trainer.validate(val_loader)
        trainer.train_epoch(train_loader)
        final_loss, _ = trainer.validate(val_loader)
        
        print(f"   Initial Val Loss: {initial_loss:.4f}")
        print(f"   Final Val Loss:   {final_loss:.4f}")
        
        if torch.isnan(torch.tensor(final_loss)):
            raise Exception("Loss is NaN!")
            
        print("   ✅ Training Loop: SUCCESS")
        
    except Exception as e:
        print(f"   ❌ Training FAILED: {e}")
        return

    # ---------------------------------------------------------
    # 4. VERIFY INFERENCE & PRICING
    # ---------------------------------------------------------
    log("4. Testing Inference & Pricing...")
    try:
        # Load the BEST model from production run (not the random one we just trained)
        ckpt_path = 'checkpoints/best_model.pt'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print("   (Loaded Production Weights for Pricing Test)")
        else:
            print("   (Using Random Weights - Pricing will be garbage but code path tested)")
            
        grid = torch.linspace(50, 200, 150).to(device)
        pricer = NeuralPricer(model, grid, epsilon=1.0, device=device)
        
        # Price
        sample_data = np.load(files[0])
        marginals = torch.from_numpy(sample_data['marginals']).float().to(device)
        
        t0 = time.time()
        price = pricer.price_asian_call(marginals, strike=100.0, num_paths=1000)
        dt = (time.time()-t0)*1000
        
        print(f"   Price: {price:.4f}")
        print(f"   Latency: {dt:.2f} ms")
        
        if np.isnan(price):
            raise Exception("Price is NaN")
            
        print("   ✅ Inference/Pricing: SUCCESS")
        
    except Exception as e:
        print(f"   ❌ Pricing FAILED: {e}")
        return

    # ---------------------------------------------------------
    # 5. CLEANUP
    # ---------------------------------------------------------
    log("5. Cleaning Up...")
    try:
        # Remove test files
        for f in files:
            os.remove(f)
        os.rmdir(test_dir)
        print("   ✅ Cleanup: SUCCESS")
    except:
        print("   ⚠️ Cleanup partial")

    print("\n" + "="*60)
    print("ALL SYSTEMS GO. PIPELINE VERIFIED.")
    print("="*60)

if __name__ == "__main__":
    final_system_check()
