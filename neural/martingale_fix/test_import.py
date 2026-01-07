import sys
import os
print("1. Starting imports")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
print("2. Path added")

try:
    import torch
    print("3. Torch imported")
except ImportError as e:
    print(f"Torch failed: {e}")

try:
    from neural.martingale_fix.architecture_fixed import ImprovedTransformerMMOT
    print("4. Architecture imported")
except ImportError as e:
    print(f"Architecture failed: {e}")
except Exception as e:
    print(f"Architecture error: {e}")

print("5. Done")
