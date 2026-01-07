
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Merging {len(args.inputs)} datasets...")
    
    all_marginals = []
    all_params = []
    all_u = []
    all_h = []
    all_drifts = []
    
    grid = None
    
    for fpath in args.inputs:
        print(f"Loading {fpath}...")
        try:
            data = np.load(fpath, allow_pickle=True)
            
            # Check grid consistency
            if grid is None:
                grid = data['grid']
            else:
                if not np.allclose(grid, data['grid']):
                    print(f"WARNING: Grid mismatch in {fpath}. Skipping.")
                    continue
            
            m = data['marginals']
            p = data['params']
            u = data['u']
            h = data['h']
            d = data['drifts']
            
            print(f"  + {len(m)} instances")
            
            all_marginals.append(m)
            all_params.append(p)
            all_u.append(u)
            all_h.append(h)
            all_drifts.append(d)
            
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
    if not all_marginals:
        print("No valid data found.")
        return

    # Concatenate
    merged_marginals = np.concatenate(all_marginals, axis=0)
    merged_params = np.concatenate(all_params, axis=0)
    merged_u = np.concatenate(all_u, axis=0)
    merged_h = np.concatenate(all_h, axis=0)
    merged_drifts = np.concatenate(all_drifts, axis=0)
    
    print(f"Total instances: {len(merged_marginals)}")
    
    np.savez_compressed(args.output,
                        marginals=merged_marginals,
                        grid=grid,
                        params=merged_params,
                        u=merged_u,
                        h=merged_h,
                        drifts=merged_drifts)
    
    print(f"Saved merged dataset to {args.output}")

if __name__ == "__main__":
    main()
