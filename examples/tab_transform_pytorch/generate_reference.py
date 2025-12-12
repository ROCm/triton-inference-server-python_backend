"""
Generate deterministic FT Transformer weights and CPU reference outputs.

This script creates:
1. ft_transformer.pt - Model weights with fixed random seed
2. reference_inputs.npz - Test inputs (categorical + continuous)
3. reference_outputs.npz - Expected outputs from CPU inference

Usage:
    python generate_reference.py [--output-dir DIR] [--num-samples N] [--seed S]

Example:
    python generate_reference.py --output-dir ./1 --num-samples 100 --seed 42
"""

import argparse
import numpy as np
import os
import torch

from tab_transformer_pytorch import FTTransformer

# Model configuration (must match model.py and config.pbtxt)
CATEGORIES = (10, 5, 6, 5, 8)  # Unique values per categorical feature
NUM_CONTINUOUS = 10  # Number of continuous features
MODEL_CONFIG = {
    "categories": CATEGORIES,
    "num_continuous": NUM_CONTINUOUS,
    "dim": 32,
    "dim_out": 1,
    "depth": 6,
    "heads": 8,
    "attn_dropout": 0.0,  # Disabled for deterministic inference
    "ff_dropout": 0.0,  # Disabled for deterministic inference
}


def generate_categorical_inputs(num_samples: int, seed: int) -> np.ndarray:
    """Generate deterministic categorical inputs."""
    rng = np.random.default_rng(seed)
    categorical_data = np.zeros((num_samples, len(CATEGORIES)), dtype=np.int64)
    for i, max_val in enumerate(CATEGORIES):
        categorical_data[:, i] = rng.integers(0, max_val, size=num_samples)
    return categorical_data


def generate_continuous_inputs(num_samples: int, seed: int) -> np.ndarray:
    """Generate deterministic continuous inputs (normalized)."""
    rng = np.random.default_rng(seed + 1000)  # Different seed for continuous
    return rng.standard_normal((num_samples, NUM_CONTINUOUS)).astype(np.float32)


def create_model(seed: int) -> FTTransformer:
    """Create FTTransformer with deterministic initialization."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = FTTransformer(**MODEL_CONFIG)
    model.eval()
    return model


def run_cpu_inference(
    model: FTTransformer,
    x_categ: np.ndarray,
    x_numer: np.ndarray,
) -> np.ndarray:
    """Run inference on CPU and return outputs."""
    model.cpu()
    model.eval()

    x_categ_tensor = torch.from_numpy(x_categ).long()
    x_numer_tensor = torch.from_numpy(x_numer).float()

    with torch.no_grad():
        output = model(x_categ_tensor, x_numer_tensor)

    return output.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Generate FT Transformer reference weights and outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./1",
        help="Output directory for weights and reference data (default: ./1)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test samples to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("FT Transformer Reference Generation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    print(f"Model config: {MODEL_CONFIG}")
    print()

    # Step 1: Create model with deterministic weights
    print("[1/4] Creating model with deterministic weights...")
    model = create_model(args.seed)

    weights_path = os.path.join(args.output_dir, "ft_transformer.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"      Saved weights to: {weights_path}")

    # Step 2: Generate deterministic inputs
    print("[2/4] Generating deterministic test inputs...")
    x_categ = generate_categorical_inputs(args.num_samples, args.seed)
    x_numer = generate_continuous_inputs(args.num_samples, args.seed)

    inputs_path = os.path.join(args.output_dir, "reference_inputs.npz")
    np.savez(inputs_path, categorical=x_categ, continuous=x_numer)
    print(f"      Saved inputs to: {inputs_path}")
    print(f"      Categorical shape: {x_categ.shape}")
    print(f"      Continuous shape: {x_numer.shape}")

    # Step 3: Run CPU inference
    print("[3/4] Running CPU inference...")
    outputs = run_cpu_inference(model, x_categ, x_numer)

    outputs_path = os.path.join(args.output_dir, "reference_outputs.npz")
    np.savez(outputs_path, outputs=outputs)
    print(f"      Saved outputs to: {outputs_path}")
    print(f"      Output shape: {outputs.shape}")

    # Step 4: Print sample results for verification
    print("[4/4] Sample results (first 5):")
    print()
    for i in range(min(5, args.num_samples)):
        print(f"      Sample {i}:")
        print(f"        Categorical: {x_categ[i]}")
        print(f"        Continuous:  {x_numer[i][:5]}... (truncated)")
        print(f"        Output:      {outputs[i][0]:.8f}")
    print()

    print("=" * 60)
    print("Reference generation complete!")
    print()
    print("Next steps:")
    print(f"  1. Copy {args.output_dir}/ to your model repository")
    print("  2. Start Triton server with the model")
    print("  3. Run: python client.py --verify --reference-dir " + args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
