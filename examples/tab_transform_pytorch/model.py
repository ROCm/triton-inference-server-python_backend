import json
import os
import torch

import triton_python_backend_utils as pb_utils

from tab_transformer_pytorch import FTTransformer


class TritonPythonModel:
    """Triton Python model for FTTransformer inference.

    This model uses the FTTransformer from tab-transformer-pytorch for
    tabular data prediction with both categorical and continuous features.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])
        self.model_repository = args["model_repository"]
        self.model_version = args["model_version"]

        # Get output configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        # Determine the device
        device_id = args["model_instance_device_id"]
        instance_kind = args["model_instance_kind"]

        if instance_kind == "GPU":
            self.device = torch.device(f"cuda:{device_id}")
            pb_utils.Logger.log_info(f"FTTransformer initialized on GPU {device_id}")
        else:
            self.device = torch.device("cpu")
            pb_utils.Logger.log_info("FTTransformer initialized on CPU")

        # FTTransformer model configuration
        # These should match your trained model's configuration
        self.categories = (10, 5, 6, 5, 8)  # Unique values per categorical feature
        self.num_continuous = 10  # Number of continuous features

        # Initialize FTTransformer model
        # Note: Dropout is disabled (0.0) to ensure deterministic inference
        # for verification against CPU reference outputs
        self.model = FTTransformer(
            categories=self.categories,
            num_continuous=self.num_continuous,
            dim=32,  # Embedding dimension (paper recommends 32)
            dim_out=1,  # Output dimension (1 for binary/regression)
            depth=6,  # Number of transformer layers (paper recommends 6)
            heads=8,  # Number of attention heads (paper recommends 8)
            attn_dropout=0.0,  # Disabled for deterministic verification
            ff_dropout=0.0,  # Disabled for deterministic verification
        )

        # Load pre-trained weights if available
        weights_path = os.path.join(
            self.model_repository, self.model_version, "ft_transformer.pt"
        )
        if os.path.exists(weights_path):
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            pb_utils.Logger.log_info(f"Loaded model weights from {weights_path}")
        else:
            pb_utils.Logger.log_warn(
                f"No weights found at {weights_path}. Using randomly initialized model."
            )

        self.model.to(self.device)
        self.model.eval()

        pb_utils.Logger.log_info(
            f"FTTransformer initialized on {self.device} with "
            f"categories={self.categories}, num_continuous={self.num_continuous}"
        )

    def execute(self, requests):
        """`execute` is called when inference is requested for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse
        """
        responses = []

        for request in requests:
            try:
                # Get categorical input (INPUT0) - shape: [batch_size, num_categories]
                input0_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
                x_categ = torch.from_numpy(input0_tensor.as_numpy()).to(self.device)

                # Get continuous input (INPUT1) - shape: [batch_size, num_continuous]
                input1_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT1")
                x_numer = torch.from_numpy(input1_tensor.as_numpy()).to(self.device)

                # Run inference
                with torch.no_grad():
                    output = self.model(x_categ, x_numer)

                # Convert output to numpy and create response tensor
                output_np = output.cpu().numpy().astype(self.output0_dtype)
                output_tensor = pb_utils.Tensor("OUTPUT0", output_np)

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )

            except Exception as e:
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Inference failed: {str(e)}"),
                )

            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.

        This function allows the model to perform any necessary clean ups
        before exit.
        """
        pb_utils.Logger.log_info("Cleaning up FTTransformer model...")
