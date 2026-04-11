"""
preprocess.py
=============
Preprocesses the Hateful Memes dataset to handle noise and improve data quality.

Issues addressed:
1. Duplicate texts with conflicting labels (897 cases)
2. Duplicate texts (1428 total duplicates)
3. Text with unusual special characters (591 cases)
4. Very short texts (<10 chars: 27 cases)
5. Class imbalance (64.1% non-hateful vs 35.9% hateful)
6. Text normalization (whitespace, encoding issues)

Usage:
    python preprocess.py --input_dir Data --output_dir Data/processed
"""

import os
import re
import argparse
import hashlib
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from PIL import Image
import io


class HatefulMemesPreprocessor:
    """Preprocesses the Hateful Memes dataset for cleaner training."""
    
    def __init__(
        self,
        remove_duplicates: bool = True,
        resolve_label_conflicts: str = "majority",  # "majority", "hateful", "remove"
        normalize_text: bool = True,
        min_text_length: int = 5,
        max_text_length: int = 500,
        validate_images: bool = True,
        verbose: bool = True
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            remove_duplicates: Remove duplicate text entries
            resolve_label_conflicts: How to handle same text with different labels
                - "majority": Use majority vote label
                - "hateful": Default to hateful (1) when conflicting
                - "remove": Remove all conflicting entries
            normalize_text: Clean and normalize text content
            min_text_length: Minimum text length to keep
            max_text_length: Maximum text length to keep
            validate_images: Check image bytes are valid
            verbose: Print progress information
        """
        self.remove_duplicates = remove_duplicates
        self.resolve_label_conflicts = resolve_label_conflicts
        self.normalize_text = normalize_text
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.validate_images = validate_images
        self.verbose = verbose
        
        self.stats = {
            "original_count": 0,
            "after_dedup": 0,
            "conflicts_resolved": 0,
            "invalid_images": 0,
            "short_texts_removed": 0,
            "long_texts_truncated": 0,
            "final_count": 0
        }
    
    def log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(f"[Preprocess] {message}")
    
    def normalize_text_content(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        - Remove excess whitespace
        - Fix common encoding issues
        - Normalize quotes and apostrophes
        - Remove zero-width characters
        - Lowercase (optional, disabled by default for sentiment)
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Normalize unicode quotes and apostrophes
        quote_replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '«': '"', '»': '"', '„': '"', '‟': '"',
            '′': "'", '″': '"', '‹': "'", '›': "'"
        }
        for old, new in quote_replacements.items():
            text = text.replace(old, new)
        
        # Normalize whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix common OCR/encoding errors
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        
        return text
    
    def validate_image(self, img_data: dict) -> bool:
        """Check if image bytes are valid and can be opened."""
        try:
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_bytes = img_data['bytes']
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                return False
            
            # Try to open the image
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()  # Verify it's a valid image
            return True
        except Exception:
            return False
    
    def compute_text_hash(self, text: str) -> str:
        """Compute hash for deduplication."""
        normalized = self.normalize_text_content(text).lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def resolve_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate texts with conflicting labels.
        
        Returns dataframe with conflicts resolved according to strategy.
        For test sets without labels, simply deduplicate by text.
        """
        # Add text hash for grouping
        df['text_hash'] = df['text'].apply(self.compute_text_hash)
        
        has_labels = 'label' in df.columns
        
        if not has_labels:
            # No labels - just deduplicate by keeping first occurrence
            df_dedup = df.drop_duplicates(subset=['text_hash'], keep='first')
            self.stats['conflicts_resolved'] = 0
            self.log(f"No labels found - deduplicated to {len(df_dedup)} unique texts")
            return df_dedup.drop(columns=['text_hash'])
        
        # Find groups with same text but different labels
        agg_dict = {
            'label': list,
            'id': list,
            'text': 'first',
            'image': 'first',
        }
        if 'img' in df.columns:
            agg_dict['img'] = 'first'
            
        grouped = df.groupby('text_hash').agg(agg_dict).reset_index()
        
        resolved_rows = []
        conflicts_count = 0
        
        for _, row in grouped.iterrows():
            labels = row['label']
            
            if len(set(labels)) > 1:
                # Conflict exists
                conflicts_count += 1
                
                if self.resolve_label_conflicts == "remove":
                    continue  # Skip this group entirely
                elif self.resolve_label_conflicts == "hateful":
                    final_label = 1
                elif self.resolve_label_conflicts == "majority":
                    # Majority vote
                    label_counts = Counter(labels)
                    final_label = label_counts.most_common(1)[0][0]
                else:
                    final_label = labels[0]
            else:
                final_label = labels[0]
            
            row_dict = {
                'id': row['id'][0],  # Keep first ID
                'text': row['text'],
                'label': final_label,
                'image': row['image'],
            }
            if 'img' in row.index:
                row_dict['img'] = row['img']
            
            resolved_rows.append(row_dict)
        
        self.stats['conflicts_resolved'] = conflicts_count
        self.log(f"Resolved {conflicts_count} label conflicts using '{self.resolve_label_conflicts}' strategy")
        
        return pd.DataFrame(resolved_rows)
    
    def process_dataframe(self, df: pd.DataFrame, split_name: str = "data") -> pd.DataFrame:
        """
        Process a single dataframe (train/val/test).
        
        Returns cleaned dataframe.
        """
        self.log(f"\n{'='*50}")
        self.log(f"Processing {split_name}: {len(df)} samples")
        self.stats['original_count'] = len(df)
        
        # Step 1: Normalize text
        if self.normalize_text:
            df = df.copy()
            df['text'] = df['text'].apply(self.normalize_text_content)
            self.log(f"Text normalized")
        
        # Step 2: Remove/resolve duplicates
        if self.remove_duplicates:
            df = self.resolve_conflicts(df)
            self.stats['after_dedup'] = len(df)
            self.log(f"After deduplication: {len(df)} samples")
        
        # Step 3: Filter by text length
        original_len = len(df)
        df = df[df['text'].str.len() >= self.min_text_length]
        self.stats['short_texts_removed'] = original_len - len(df)
        if self.stats['short_texts_removed'] > 0:
            self.log(f"Removed {self.stats['short_texts_removed']} samples with text < {self.min_text_length} chars")
        
        # Step 4: Truncate long texts
        long_mask = df['text'].str.len() > self.max_text_length
        self.stats['long_texts_truncated'] = long_mask.sum()
        if self.stats['long_texts_truncated'] > 0:
            df.loc[long_mask, 'text'] = df.loc[long_mask, 'text'].str[:self.max_text_length]
            self.log(f"Truncated {self.stats['long_texts_truncated']} texts > {self.max_text_length} chars")
        
        # Step 5: Validate images
        if self.validate_images:
            valid_mask = df['image'].apply(self.validate_image)
            invalid_count = (~valid_mask).sum()
            self.stats['invalid_images'] = invalid_count
            if invalid_count > 0:
                df = df[valid_mask]
                self.log(f"Removed {invalid_count} samples with invalid images")
        
        # Remove helper columns
        if 'text_hash' in df.columns:
            df = df.drop(columns=['text_hash'])
        
        self.stats['final_count'] = len(df)
        
        # Print summary
        self.log(f"\nSummary for {split_name}:")
        self.log(f"  Original: {self.stats['original_count']}")
        self.log(f"  Final: {self.stats['final_count']}")
        self.log(f"  Removed: {self.stats['original_count'] - self.stats['final_count']}")
        
        # Label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            total = len(df)
            self.log(f"  Label 0 (non-hateful): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")
            self.log(f"  Label 1 (hateful): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")
        
        return df.reset_index(drop=True)


def restructure_dataset(input_dir: str, output_dir: str, preprocessor: HatefulMemesPreprocessor):
    """
    Restructure the dataset with cleaner organization.
    
    New structure:
    output_dir/
    ├── train.parquet          # Cleaned training data
    ├── validation.parquet     # Cleaned validation data
    ├── test.parquet           # Cleaned test data
    ├── metadata.json          # Processing statistics
    └── sample_images/         # Sample images for inspection
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    train_file = list(input_path.glob("train*.parquet"))
    val_file = list(input_path.glob("validation*.parquet"))
    test_files = list((input_path / "test").glob("*.parquet")) if (input_path / "test").exists() else []
    
    all_stats = {}
    
    # Process train
    if train_file:
        print(f"\n{'='*60}")
        print("PROCESSING TRAINING DATA")
        print(f"{'='*60}")
        train_df = pd.read_parquet(train_file[0])
        train_clean = preprocessor.process_dataframe(train_df, "train")
        train_clean.to_parquet(output_path / "train.parquet", index=False)
        all_stats['train'] = preprocessor.stats.copy()
    
    # Process validation
    if val_file:
        print(f"\n{'='*60}")
        print("PROCESSING VALIDATION DATA")
        print(f"{'='*60}")
        val_df = pd.read_parquet(val_file[0])
        val_clean = preprocessor.process_dataframe(val_df, "validation")
        val_clean.to_parquet(output_path / "validation.parquet", index=False)
        all_stats['validation'] = preprocessor.stats.copy()
    
    # Process test
    if test_files:
        print(f"\n{'='*60}")
        print("PROCESSING TEST DATA")
        print(f"{'='*60}")
        test_df = pd.read_parquet(test_files[0])
        test_clean = preprocessor.process_dataframe(test_df, "test")
        test_clean.to_parquet(output_path / "test.parquet", index=False)
        all_stats['test'] = preprocessor.stats.copy()
    
    # Save metadata (convert numpy types to native Python types)
    import json
    
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    metadata = {
        "preprocessing_config": {
            "remove_duplicates": preprocessor.remove_duplicates,
            "resolve_label_conflicts": preprocessor.resolve_label_conflicts,
            "normalize_text": preprocessor.normalize_text,
            "min_text_length": preprocessor.min_text_length,
            "max_text_length": preprocessor.max_text_length,
            "validate_images": preprocessor.validate_images
        },
        "statistics": convert_to_native(all_stats)
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Files created:")
    for f in output_path.glob("*"):
        print(f"  - {f.name}")
    
    return all_stats


def create_balanced_dataset(
    df: pd.DataFrame,
    strategy: str = "undersample",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a balanced dataset for training.
    
    Args:
        df: Input dataframe with 'label' column
        strategy: "undersample" (reduce majority) or "oversample" (increase minority)
        random_state: Random seed for reproducibility
    
    Returns:
        Balanced dataframe
    """
    label_counts = df['label'].value_counts()
    minority_count = label_counts.min()
    majority_count = label_counts.max()
    
    if strategy == "undersample":
        # Undersample majority class
        balanced_dfs = []
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            if len(label_df) > minority_count:
                label_df = label_df.sample(n=minority_count, random_state=random_state)
            balanced_dfs.append(label_df)
        return pd.concat(balanced_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    elif strategy == "oversample":
        # Oversample minority class
        balanced_dfs = []
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            if len(label_df) < majority_count:
                # Sample with replacement to reach majority count
                label_df = label_df.sample(n=majority_count, replace=True, random_state=random_state)
            balanced_dfs.append(label_df)
        return pd.concat(balanced_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Hateful Memes dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with default settings
  python preprocess.py --input_dir Data --output_dir Data/processed
  
  # Remove all conflicting labels instead of majority vote
  python preprocess.py --input_dir Data --output_dir Data/processed --resolve_conflicts remove
  
  # Keep duplicates (no deduplication)
  python preprocess.py --input_dir Data --output_dir Data/processed --keep_duplicates
  
  # Create balanced dataset
  python preprocess.py --input_dir Data --output_dir Data/processed --balance undersample
"""
    )
    
    parser.add_argument(
        "--input_dir", type=str, default="Data",
        help="Input directory containing parquet files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="Data/processed",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--resolve_conflicts", type=str, default="majority",
        choices=["majority", "hateful", "remove"],
        help="How to resolve conflicting labels (default: majority)"
    )
    parser.add_argument(
        "--keep_duplicates", action="store_true",
        help="Keep duplicate texts (no deduplication)"
    )
    parser.add_argument(
        "--min_text_length", type=int, default=5,
        help="Minimum text length to keep (default: 5)"
    )
    parser.add_argument(
        "--max_text_length", type=int, default=500,
        help="Maximum text length (truncate longer, default: 500)"
    )
    parser.add_argument(
        "--skip_image_validation", action="store_true",
        help="Skip image validation (faster but may keep corrupt images)"
    )
    parser.add_argument(
        "--balance", type=str, default=None,
        choices=["undersample", "oversample"],
        help="Balance the dataset (optional)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = HatefulMemesPreprocessor(
        remove_duplicates=not args.keep_duplicates,
        resolve_label_conflicts=args.resolve_conflicts,
        normalize_text=True,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        validate_images=not args.skip_image_validation,
        verbose=not args.quiet
    )
    
    # Run preprocessing
    stats = restructure_dataset(args.input_dir, args.output_dir, preprocessor)
    
    # Optional: Balance the training set
    if args.balance:
        print(f"\n{'='*60}")
        print(f"BALANCING TRAINING DATA ({args.balance})")
        print(f"{'='*60}")
        
        output_path = Path(args.output_dir)
        train_df = pd.read_parquet(output_path / "train.parquet")
        
        original_dist = train_df['label'].value_counts().to_dict()
        print(f"Original distribution: {original_dist}")
        
        balanced_df = create_balanced_dataset(train_df, strategy=args.balance)
        balanced_dist = balanced_df['label'].value_counts().to_dict()
        print(f"Balanced distribution: {balanced_dist}")
        
        # Save balanced version
        balanced_df.to_parquet(output_path / "train_balanced.parquet", index=False)
        print(f"Saved balanced training set to {output_path / 'train_balanced.parquet'}")


if __name__ == "__main__":
    main()
