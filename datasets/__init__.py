# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig
from .customdata import CustomDetectionDataset, CustomDatasetConfig  # Add this import
import os
import glob
DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "custom": [CustomDetectionDataset,CustomDatasetConfig]  # Add your custom dataset
}

def build_dataset(args):
  print("\n=== DEBUGGING DATASET CREATION ===")
  print(f"Root dir: {args.dataset_root_dir}")
  print(args.dataset_root_dir)
  
  # Recursively find all pc.npy files in subdirectories
  all_scenes = glob.glob(os.path.join(args.dataset_root_dir, "**/pc.npy"), recursive=True)
  
  # Extract relative paths (e.g., "subdir/pc.npy" instead of full paths)
  all_scenes = [os.path.relpath(scene, args.dataset_root_dir) for scene in all_scenes]
  
  print(f"\nFound {len(all_scenes)} scene files")
  print(f"Sample files: {all_scenes[:3]}...")
  
  if not all_scenes:
      raise ValueError("No matching scene files found! Check:")
      print("- File extensions (must be 'pc.npy')")
      print("- Directory structure (files should be in subdirectories)")
  
  dataset_config = CustomDatasetConfig()

  dataset_dict = {}
  for split in ["train", "test"]:
      print(f"\nBuilding {split} split...")
      dataset = CustomDetectionDataset(
          config=dataset_config,
          split_set=split,
          root_dir=args.dataset_root_dir,
          all_scenes = all_scenes,
          use_color=args.use_color,
          augment=(split == "train")
      )
      
      print(f"{split} dataset contains {len(dataset)} samples")
      if len(dataset) == 0:
          print("WARNING: Empty dataset! Checking:")
          print(f"- Total scenes: {len(all_scenes)}")
          print(f"- Split ratio: 80/20")
          print(f"- Shuffle seed: {dataset.rng_seed if hasattr(dataset, 'rng_seed') else 'N/A'}")
          
      dataset_dict[split] = dataset
  
  print("\n=== DATASET BUILD COMPLETE ===")
  return dataset_dict, dataset_config

