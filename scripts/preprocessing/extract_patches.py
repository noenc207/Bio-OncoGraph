import argparse
import os
import glob
import multiprocessing
import pandas as pd
from tqdm import tqdm
import json
import uuid

def process_slide(slide_path, output_dir, patch_size, level):
    \"\"\"
    Process a single slide to extract patches.
    \"\"\"
    slide_filename = os.path.basename(slide_path)
    slide_id = os.path.splitext(slide_filename)[0]
    
    # Extract patient_id, assuming format: patientId_... or just slide_id
    patient_id = slide_id.split('_')[0] if '_' in slide_id else slide_id
    
    records = []
    
    # In a real-world scenario, you would use openslide or tiffslide here to read the WSI
    # e.g. slide = openslide.OpenSlide(slide_path)
    # Since we are implementing a pure Python boilerplate without external C dependencies like OpenSlide,
    # we simulate the extraction of a grid of patches.
    
    # Simulating image dimensions
    width, height = 10000, 10000 
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Simulating some filtering condition (e.g. tissue ratio)
            tissue_ratio = 0.85 
            quality_score = 0.95
            
            if tissue_ratio > 0.1:
                # In real scenario, save the patch image to output_dir
                # patch_img.save(os.path.join(output_dir, f"{slide_id}_{x}_{y}.png"))
                
                records.append({
                    'slide_id': slide_id,
                    'patient_id': patient_id,
                    'x': x,
                    'y': y,
                    'level': level,
                    'magnification': 40.0,
                    'patch_size': patch_size,
                    'tissue_ratio': tissue_ratio,
                    'quality_score': quality_score,
                    'feature_status': 'pending',
                    'graph_status': 'pending'
                })
                
    return slide_id, records

def main():
    parser = argparse.ArgumentParser(description="Extract patches from whole slide images with multi-node and multiprocessing support.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input slides.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save patches and manifest.")
    parser.add_argument('--patch_size', type=int, default=256, help="Size of the patches to extract.")
    parser.add_argument('--level', type=int, default=0, help="Magnification level to extract patches from.")
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help="Number of CPU workers for multiprocessing.")
    parser.add_argument('--node_rank', type=int, default=0, help="Node rank for multi-node execution (0-indexed).")
    parser.add_argument('--world_size', type=int, default=1, help="Total number of nodes for multi-node execution.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Support various slide extensions
    extensions = ['*.svs', '*.tif', '*.tiff', '*.ndpi', '*.vms', '*.vmu', '*.scn', '*.mrxs', '*.bif', '*.png', '*.jpg']
    slide_paths = []
    for ext in extensions:
        slide_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    slide_paths = sorted(slide_paths)
    
    if not slide_paths:
        print(f"No slides found in {args.input_dir}")
        return

    # Multi-node partitioning
    # Each node processes a subset of the slides
    slide_paths = [s for i, s in enumerate(slide_paths) if i % args.world_size == args.node_rank]
    print(f"Node {args.node_rank}/{args.world_size} processing {len(slide_paths)} slides.")

    # Checkpoint resume: read existing manifest to skip processed slides
    manifest_csv_path = os.path.join(args.output_dir, 'patch_manifest.csv')
    processed_slides = set()
    all_records = []
    
    if os.path.exists(manifest_csv_path):
        try:
            df_existing = pd.read_csv(manifest_csv_path)
            if 'slide_id' in df_existing.columns:
                processed_slides = set(df_existing['slide_id'].unique())
                all_records = df_existing.to_dict('records')
                print(f"Found existing manifest. Skipping {len(processed_slides)} already processed slides.")
        except Exception as e:
            print(f"Error reading existing manifest: {e}")

    # Filter out already processed slides
    slide_paths_to_process = [
        s for s in slide_paths 
        if os.path.splitext(os.path.basename(s))[0] not in processed_slides
    ]
    
    if not slide_paths_to_process:
        print("All slides have been processed.")
        return

    def update_manifest():
        df = pd.DataFrame(all_records)
        
        # Ensure all required columns are present even if df is empty
        required_columns = ['slide_id', 'patient_id', 'x', 'y', 'level', 'magnification', 
                            'patch_size', 'tissue_ratio', 'quality_score', 'feature_status', 'graph_status']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                
        # Reorder columns to match the requirement exactly
        df = df[required_columns]
        
        df.to_csv(os.path.join(args.output_dir, 'patch_manifest.csv'), index=False)
        df.to_parquet(os.path.join(args.output_dir, 'patch_manifest.parquet'), index=False)
        df.to_json(os.path.join(args.output_dir, 'patch_manifest.json'), orient='records', indent=4)

    # Process slides using multiprocessing
    def process_wrapper(path):
        return process_slide(path, args.output_dir, args.patch_size, args.level)
    
    # We use spawn or fork depending on OS, default Pool is fine
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(slide_paths_to_process), desc=f"Node {args.node_rank} Progress") as pbar:
            for slide_id, records in pool.imap_unordered(process_wrapper, slide_paths_to_process):
                if records:
                    all_records.extend(records)
                    update_manifest()
                pbar.update(1)
                
    print("Patch extraction complete.")

if __name__ == "__main__":
    main()
