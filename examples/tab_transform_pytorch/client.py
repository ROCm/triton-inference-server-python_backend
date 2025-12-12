"""
Client for testing FTTransformer model on Triton Inference Server.

Usage:
    # Random inference test
    python client.py [--url URL] [--batch-size N] [--verbose]

    # Verification mode (compare GPU outputs against CPU reference)
    python client.py --verify --reference-dir ./1 [--url URL] [--tolerance T]

Example:
    python client.py --url localhost:8000 --batch-size 2 --verbose
    python client.py --verify --reference-dir ./1 --tolerance 1e-5
"""

import argparse
import os
import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

MODEL_NAME = "tab_transform_pytorch"

# Model configuration (should match model.py and config.pbtxt)
NUM_CATEGORIES = 5  # Number of categorical features
NUM_CONTINUOUS = 10  # Number of continuous features
CATEGORIES = (10, 5, 6, 5, 8)  # Max unique values per categorical feature


def generate_categorical_input(batch_size: int) -> np.ndarray:
    """Generate random categorical input data.

    Each categorical feature value must be in range [0, max_category_value).
    """
    categorical_data = np.zeros((batch_size, NUM_CATEGORIES), dtype=np.int64)
    for i, max_val in enumerate(CATEGORIES):
        categorical_data[:, i] = np.random.randint(0, max_val, size=batch_size)
    return categorical_data


def generate_continuous_input(batch_size: int) -> np.ndarray:
    """Generate random continuous input data (normalized)."""
    return np.random.randn(batch_size, NUM_CONTINUOUS).astype(np.float32)


def run_inference(
    client: httpclient.InferenceServerClient,
    categorical_data: np.ndarray,
    continuous_data: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Run inference and return output array."""
    if verbose:
        print(f"\nCategorical input (INPUT0):\n{categorical_data}")
        print(f"\nContinuous input (INPUT1):\n{continuous_data}")

    # Prepare inputs
    inputs = [
        httpclient.InferInput(
            "INPUT0",
            categorical_data.shape,
            np_to_triton_dtype(categorical_data.dtype),
        ),
        httpclient.InferInput(
            "INPUT1",
            continuous_data.shape,
            np_to_triton_dtype(continuous_data.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(categorical_data)
    inputs[1].set_data_from_numpy(continuous_data)

    # Prepare outputs
    outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

    # Run inference
    response = client.infer(MODEL_NAME, inputs, request_id=str(1), outputs=outputs)

    output = response.as_numpy("OUTPUT0")
    if output is None:
        raise RuntimeError("No output data received from server")
    return output


def run_random_inference(url: str, batch_size: int, verbose: bool = False) -> bool:
    """Run inference with random inputs (original behavior).

    Args:
        url: Triton server URL (e.g., "localhost:8000")
        batch_size: Number of samples in the batch
        verbose: Whether to print detailed output

    Returns:
        True if inference succeeded, False otherwise
    """
    try:
        with httpclient.InferenceServerClient(url, verbose=verbose) as client:
            # Check if model is ready
            if not client.is_model_ready(MODEL_NAME):
                print(f"ERROR: Model '{MODEL_NAME}' is not ready on server")
                return False

            # Generate input data
            categorical_data = generate_categorical_input(batch_size)
            continuous_data = generate_continuous_input(batch_size)

            # Run inference
            output_data = run_inference(
                client, categorical_data, continuous_data, verbose
            )

            if output_data is None:
                print("ERROR: No output data received from server")
                return False

            print("\n" + "=" * 60)
            print("FTTransformer Inference Results")
            print("=" * 60)
            print(f"Batch size: {batch_size}")
            print("Input shapes:")
            print(f"  - Categorical (INPUT0): {categorical_data.shape}")
            print(f"  - Continuous  (INPUT1): {continuous_data.shape}")
            print(f"Output shape: {output_data.shape}")
            print("\nPredictions (OUTPUT0):")
            for i, pred in enumerate(output_data):
                print(f"  Sample {i}: {pred}")
            print("=" * 60)

            # Basic validation: output shape should be (batch_size, 1)
            expected_shape = (batch_size, 1)
            if output_data.shape != expected_shape:
                print(
                    f"ERROR: Unexpected output shape. "
                    f"Expected {expected_shape}, got {output_data.shape}"
                )
                return False

            print("\nPASS: tab_transform_pytorch")
            return True

    except Exception as e:
        print(f"ERROR: Inference failed with exception: {e}")
        return False


def run_verification(
    url: str,
    reference_dir: str,
    tolerance: float = 1e-5,
    verbose: bool = False,
) -> bool:
    """Run verification against CPU reference outputs.

    Args:
        url: Triton server URL
        reference_dir: Directory containing reference_inputs.npz and reference_outputs.npz
        tolerance: Maximum allowed absolute difference (default: 1e-5)
        verbose: Whether to print detailed output

    Returns:
        True if verification passed, False otherwise
    """
    # Load reference data
    inputs_path = os.path.join(reference_dir, "reference_inputs.npz")
    outputs_path = os.path.join(reference_dir, "reference_outputs.npz")

    if not os.path.exists(inputs_path):
        print(f"ERROR: Reference inputs not found: {inputs_path}")
        print("       Run generate_reference.py first")
        return False

    if not os.path.exists(outputs_path):
        print(f"ERROR: Reference outputs not found: {outputs_path}")
        print("       Run generate_reference.py first")
        return False

    print("=" * 60)
    print("FTTransformer Verification Mode")
    print("=" * 60)
    print(f"Reference directory: {reference_dir}")
    print(f"Tolerance: {tolerance}")
    print()

    # Load reference data
    print("[1/4] Loading reference data...")
    inputs_data = np.load(inputs_path)
    outputs_data = np.load(outputs_path)

    x_categ = inputs_data["categorical"]
    x_numer = inputs_data["continuous"]
    reference_outputs = outputs_data["outputs"]

    num_samples = x_categ.shape[0]
    print(f"      Loaded {num_samples} samples")
    print(f"      Categorical shape: {x_categ.shape}")
    print(f"      Continuous shape: {x_numer.shape}")
    print(f"      Reference output shape: {reference_outputs.shape}")

    try:
        with httpclient.InferenceServerClient(url, verbose=verbose) as client:
            # Check if model is ready
            print("[2/4] Checking model status...")
            if not client.is_model_ready(MODEL_NAME):
                print(f"ERROR: Model '{MODEL_NAME}' is not ready on server")
                return False
            print(f"      Model '{MODEL_NAME}' is ready")

            # Run inference in batches (max_batch_size is 4)
            print("[3/4] Running GPU inference...")
            batch_size = 4
            gpu_outputs = []

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_categ = x_categ[start_idx:end_idx]
                batch_numer = x_numer[start_idx:end_idx]

                output = run_inference(client, batch_categ, batch_numer, verbose=False)
                gpu_outputs.append(output)

                if verbose:
                    print(f"      Processed samples {start_idx}-{end_idx}")

            gpu_outputs = np.vstack(gpu_outputs)
            print(f"      GPU output shape: {gpu_outputs.shape}")

            # Compare outputs
            print("[4/4] Comparing outputs...")
            abs_diff = np.abs(gpu_outputs - reference_outputs)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            num_mismatches = np.sum(abs_diff > tolerance)

            print()
            print("Results:")
            print(f"  Max absolute difference:  {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Tolerance:                {tolerance:.2e}")
            print(f"  Samples exceeding tolerance: {num_mismatches}/{num_samples}")

            if verbose or num_mismatches > 0:
                print()
                print("Sample-by-sample comparison (first 10):")
                for i in range(min(10, num_samples)):
                    diff = abs_diff[i][0]
                    status = "✓" if diff <= tolerance else "✗"
                    print(
                        f"  [{status}] Sample {i:3d}: "
                        f"CPU={reference_outputs[i][0]:+.8f}, "
                        f"GPU={gpu_outputs[i][0]:+.8f}, "
                        f"diff={diff:.2e}"
                    )

            print()
            print("=" * 60)

            if max_diff <= tolerance:
                print(
                    f"PASS: All {num_samples} samples within tolerance ({tolerance:.0e})"
                )
                print("      GPU implementation matches CPU reference!")
                print("=" * 60)
                return True
            else:
                print(
                    f"FAIL: {num_mismatches} samples exceed tolerance ({tolerance:.0e})"
                )
                print("      GPU implementation may have numerical issues")
                print("=" * 60)
                return False

    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Client for FTTransformer model on Triton Inference Server"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for random inference (default: 1, max: 4 per config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification mode (compare against CPU reference)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="./1",
        help="Directory containing reference data (default: ./1)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Max absolute difference allowed in verification (default: 1e-5)",
    )

    args = parser.parse_args()

    if args.verify:
        success = run_verification(
            args.url,
            args.reference_dir,
            args.tolerance,
            args.verbose,
        )
    else:
        success = run_random_inference(args.url, args.batch_size, args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
