"""
pb4_absa_inference.py — Aspect-Based Sentiment Analysis (ABSA) Inference Module

This script is a refactored and professionally adapted local version
of the Google Colab environment notebook. It performs a full ABSA analysis using
the ATEPC (Aspect Term Extraction and Polarity Classification) model from the PyABSA library.

The script expects the input file 'data/03_processed/absa_input.csv' and
returns the processed results to 'data/04_results/absa_results.csv'.
"""

import os
import sys
import pandas as pd
import warnings

# ==============================================================================
# MONKEY PATCH: Environment fix for DeBERTa model configuration
# Solves the decoder parameter conflict problem occurring in newer versions
# of the transformers library working with PyABSA on the PyTorch backend.
# ==============================================================================
import transformers
try:
    from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
    DebertaV2Config.is_decoder = False
except ImportError:
    warnings.warn("Failed to apply patch to DebertaV2Config - the library might not be fully installed.")
# ==============================================================================

# Block unnecessary warning logs from the transformers library
warnings.filterwarnings('ignore')


def run_absa_extraction(input_path: str, output_path: str):
    """
    Runs the ABSA model on the given dataset and exports a flat results table.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)
        
    print("=" * 60)
    print("1. Loading dataset for aspect analysis...")
    print("=" * 60)
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reviews.")

    # Delayed import due to library initialization time and patch
    from pyabsa import AspectTermExtraction as ATEPC

    print("\n2. Initializing ATEPC model (Aspect Extractor)...")
    # The model defaults to the English language engine and auto-detects the GPU (CUDA/MPS/CPU)
    aspect_extractor = ATEPC.AspectExtractor('english', auto_device=True)

    print("\n3. Starting ABSA inference (may take a long time on CPU)...")
    # Protection against missing values in the analyzed text
    texts = df['neg_text'].dropna().astype(str).tolist()

    # Call the ABSA model on the list of texts
    results = aspect_extractor.extract_aspect(
        inference_source=texts,
        pred_sentiment=True
    )

    print("\n4. Structuring and flattening result lists into a dataframe...")
    absa_records = []

    # Map the extracted result lists back to the unique review IDs
    for i, res in enumerate(results):
        if not res.get('aspect'):
            continue

        original_id = df.iloc[i]['source_id']
        hotel_stars = df.iloc[i]['hotel_stars']
        traveler_type = df.iloc[i]['traveler_type']

        for aspect, sentiment in zip(res['aspect'], res['sentiment']):
            absa_records.append({
                'source_id': original_id,
                'hotel_stars': hotel_stars,
                'traveler_type': traveler_type,
                'extracted_aspect': aspect.lower(),
                'sentiment': sentiment.upper()
            })

    df_results = pd.DataFrame(absa_records)
    
    # Create the output path if the folders are missing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding='utf-8')

    print("=" * 60)
    print(f"[SUCCESS] Analysis complete. Extracted {len(df_results)} single aspects.")
    print(f"[SAVED] Results exported to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Standard paths in the project repository architecture
    INPUT_FILE = "data/03_processed/absa_input.csv"
    OUTPUT_FILE = "data/04_results/absa_results.csv"
    
    run_absa_extraction(INPUT_FILE, OUTPUT_FILE)
