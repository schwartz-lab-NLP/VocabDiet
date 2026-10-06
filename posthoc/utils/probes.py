import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Optional


class FlexProbe(nn.Module):
    """
    A flexible probe for token classification that supports:
    1. Multiple representative directions per category
    2. Either max-pooling or MLP aggregation of representative scores
    3. Optional normalization of representative directions
    4. Handling of classes with no representatives
    5. Special handling of negative classes with zero vectors that aren't updated
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        use_mlp: bool = False,
        mlp_hidden_dims: List[int] = None,
        activation: nn.Module = None,
        empty_class_value: float = -18.0,
        normalize_reps: bool = False,
        normalize_temperature: float = 20.0,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the FlexProbe module structure.
        Call set_representatives() after initialization to set the actual representative values.
        Optionally call set_negative_classes() and set_classes_to_merge() to configure class behavior.

        Args:
            hidden_size: Dimension of input hidden states
            num_classes: Number of classes
            use_mlp: If True, use MLP instead of max-pooling for aggregating representative scores
            mlp_hidden_dims: List of hidden dimensions for MLP layers (only used if use_mlp=True)
            activation: Activation function to use in MLP (defaults to SiLU)
            empty_class_value: Value to use for classes with no representatives
            normalize_reps: Whether to normalize representative directions with RMS norm
            normalize_temperature: Temperature for RMS normalization
            dtype: Data type for parameters
        """
        super().__init__()

        # Store configuration
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.normalize_reps = normalize_reps
        self.normalize_temperature = normalize_temperature
        self.use_mlp = use_mlp
        self.empty_class_value = empty_class_value
        self.dtype = dtype

        # Initialize as empty - can be set later
        self.negative_class_ids = set()
        self.classes_to_merge = []

        # Initialize placeholders - will be set by set_representatives()
        self.normal_representatives = nn.Parameter(torch.empty(0, hidden_size, dtype=dtype))
        self.register_buffer("negative_representatives", torch.empty(0, hidden_size, dtype=dtype))

        # Class organization - will be populated by set_representatives()
        self.classes_with_reps = []
        self.negative_classes = []
        self.empty_classes = list(range(num_classes))
        self.class_to_rep_indices = {}
        self.class_to_rep_type = {}
        self.has_representatives = False

        # Setup MLP if requested
        self._setup_mlp(activation or nn.SiLU())

    def _setup_mlp(self, activation: nn.Module):
        """Create optional MLP layers for aggregating transformation scores."""
        if not self.use_mlp:
            self.mlp = None
            return

        hidden_dims = self.mlp_hidden_dims or [max(self.num_classes // 2, 128)]
        layers = []
        input_dim = self.num_classes
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim, bias=False), activation])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.num_classes, bias=False))
        self.mlp = nn.Sequential(*layers)

    def set_representatives(
        self,
        representatives: Union[
            Dict[int, torch.Tensor], List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor
        ],
    ):
        """
        Set the representative vectors after initialization.

        Args:
            representatives: Can be one of:
                - Dict mapping class_idx -> tensor of shape [num_reps, hidden_size] or [hidden_size]
                - List of lists where each sublist contains tensors of shape [hidden_size]
                - List of tensors where each tensor is shape [num_reps, hidden_size] or [hidden_size]
                - Single tensor of shape [num_classes, hidden_size] (one rep per class)
                Empty lists or tensors of shape [0, hidden_size] indicate classes with no representatives
        """
        # Handle single tensor case (one representative per class)
        if isinstance(representatives, torch.Tensor):
            if representatives.dim() == 2 and representatives.shape[0] <= self.num_classes:
                # Convert to dict format
                representatives = {
                    i: representatives[i : i + 1] for i in range(representatives.shape[0])
                }
            else:
                raise ValueError(
                    f"Single tensor must have shape [num_classes, hidden_size], got {representatives.shape}"
                )

        # Store the representatives for potential reorganization later
        self._last_representatives = representatives
        self._representatives_set = True

        # Convert to standardized dict format
        rep_dict = self._standardize_representatives(representatives)

        # Organize classes by type
        self._organize_classes(rep_dict)

        # Count representatives
        total_normal_reps = sum(
            rep_dict[class_idx].shape[0] for class_idx in self.classes_with_reps
        )
        total_negative_reps = len(self.negative_classes)

        # Create and populate parameter tensors
        self._create_representative_tensors(rep_dict, total_normal_reps, total_negative_reps)

        # Apply normalization if requested
        if self.normalize_reps and self.normal_representatives.shape[0] > 0:
            self._apply_normalization()

        self.has_representatives = len(self.classes_with_reps) > 0 or len(self.negative_classes) > 0

    def _standardize_representatives(
        self,
        representatives: Union[
            Dict[int, torch.Tensor], List[List[torch.Tensor]], List[torch.Tensor]
        ],
    ) -> Dict[int, torch.Tensor]:
        """Convert representatives to standardized dict format."""
        if isinstance(representatives, dict):
            # Ensure all dict values are properly formatted tensors
            standardized = {}
            for class_idx, reps in representatives.items():
                if isinstance(reps, torch.Tensor):
                    # Already a tensor - ensure correct shape and dtype
                    if reps.dim() == 1:
                        # Single vector - add batch dimension
                        standardized[class_idx] = reps.unsqueeze(0).to(dtype=self.dtype)
                    else:
                        # Multiple vectors - use as is
                        standardized[class_idx] = reps.to(dtype=self.dtype)
                elif isinstance(reps, (list, tuple)):
                    if len(reps) == 0:
                        standardized[class_idx] = torch.empty(0, self.hidden_size, dtype=self.dtype)
                    else:
                        # Convert list/tuple of tensors to single tensor
                        standardized[class_idx] = torch.stack(reps).to(dtype=self.dtype)
                else:
                    raise ValueError(
                        f"Invalid representative format for class {class_idx}: {type(reps)}"
                    )
            return standardized

        # Handle list format
        rep_dict = {}
        for class_idx, reps in enumerate(representatives):
            if isinstance(reps, torch.Tensor):
                # Single tensor for this class
                if reps.dim() == 1:
                    # Single vector - add batch dimension
                    rep_dict[class_idx] = reps.unsqueeze(0).to(dtype=self.dtype)
                else:
                    # Multiple vectors - use as is
                    rep_dict[class_idx] = reps.to(dtype=self.dtype)
            elif isinstance(reps, (list, tuple)):
                if len(reps) == 0:
                    rep_dict[class_idx] = torch.empty(0, self.hidden_size, dtype=self.dtype)
                else:
                    # Check if elements are tensors
                    if all(isinstance(r, torch.Tensor) for r in reps):
                        rep_dict[class_idx] = torch.stack(reps).to(dtype=self.dtype)
                    else:
                        raise ValueError(
                            f"All elements in representative list for class {class_idx} must be tensors"
                        )
            else:
                raise ValueError(
                    f"Invalid representative format for class {class_idx}: {type(reps)}"
                )

        return rep_dict

    def _organize_classes(self, rep_dict: Dict[int, torch.Tensor]):
        """Organize classes into different categories based on their representatives."""
        self.classes_with_reps = []
        self.negative_classes = []
        self.empty_classes = []

        for class_idx in range(self.num_classes):
            if class_idx in rep_dict and rep_dict[class_idx].shape[0] > 0:
                if class_idx in self.negative_class_ids:
                    self.negative_classes.append(class_idx)
                else:
                    self.classes_with_reps.append(class_idx)
            else:
                self.empty_classes.append(class_idx)

    def _create_representative_tensors(
        self, rep_dict: Dict[int, torch.Tensor], total_normal_reps: int, total_negative_reps: int
    ):
        """Create and populate the parameter tensors for representatives."""
        # Create normal representatives parameter
        if total_normal_reps > 0:
            # Create new parameter with correct size
            new_normal_reps = torch.zeros(total_normal_reps, self.hidden_size, dtype=self.dtype)

            # Populate with representative data
            start_idx = 0
            for class_idx in self.classes_with_reps:
                reps = rep_dict[class_idx]
                num_reps = reps.shape[0]
                new_normal_reps[start_idx : start_idx + num_reps] = reps
                self.class_to_rep_indices[class_idx] = (start_idx, start_idx + num_reps)
                self.class_to_rep_type[class_idx] = "normal"
                start_idx += num_reps

            # Replace the parameter
            self.normal_representatives = nn.Parameter(new_normal_reps)
        else:
            self.normal_representatives = nn.Parameter(
                torch.empty(0, self.hidden_size, dtype=self.dtype)
            )

        # Create negative representatives buffer (always zeros, not trainable)
        if total_negative_reps > 0:
            new_negative_reps = torch.zeros(total_negative_reps, self.hidden_size, dtype=self.dtype)

            start_idx = 0
            for class_idx in self.negative_classes:
                self.class_to_rep_indices[class_idx] = (start_idx, start_idx + 1)
                self.class_to_rep_type[class_idx] = "negative"
                start_idx += 1

            # Update the buffer
            self.negative_representatives = new_negative_reps
        else:
            self.negative_representatives = torch.empty(0, self.hidden_size, dtype=self.dtype)

    def _apply_normalization(self):
        """Apply RMS normalization to normal representatives."""
        with torch.no_grad():
            rms_norm = self.normalize_temperature * torch.sqrt(
                torch.mean(self.normal_representatives**2, dim=1, keepdim=True) + 1e-8
            )
            self.normal_representatives.data /= rms_norm

    def _merge_classes(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply class merging by taking max within each group."""
        for group in self.classes_to_merge:
            group_tensor = logits[:, group]
            max_vals, _ = group_tensor.max(dim=1, keepdim=True)
            logits[:, group] = max_vals
        return logits

    def forward(self, hidden_states: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Forward pass of the probe.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
                          or [batch_size, hidden_size]
            debug: If True, print debugging information

        Returns:
            logits: Tensor of shape [batch_size, seq_len, num_classes]
                   or [batch_size, num_classes]
        """
        if debug:
            print(
                f"Input stats: mean={hidden_states.mean():.3f}, std={hidden_states.std():.3f}, max={hidden_states.abs().max():.3f}"
            )
            if self.normal_representatives.shape[0] > 0:
                print(
                    f"Rep stats: mean={self.normal_representatives.mean():.3f}, std={self.normal_representatives.std():.3f}, max={self.normal_representatives.abs().max():.3f}"
                )

        # Handle both sequence and single token inputs
        orig_shape = hidden_states.shape
        if len(orig_shape) == 3:
            batch_size, seq_len = orig_shape[0], orig_shape[1]
            hidden_states = hidden_states.view(-1, self.hidden_size)
        else:
            batch_size, seq_len = orig_shape[0], None

        # Initialize logits with empty class values
        logits = torch.full(
            (hidden_states.shape[0], self.num_classes),
            self.empty_class_value,
            device=hidden_states.device,
            dtype=self.dtype,
        )

        # Compute scores if we have representatives
        if self.has_representatives:
            self._compute_representative_scores(hidden_states, logits)

        # Apply class merging if specified
        if self.classes_to_merge:
            logits = self._merge_classes(logits)

        # Reshape to original dimensions if necessary
        if seq_len is not None:
            logits = logits.view(batch_size, seq_len, self.num_classes)

        return logits

    def _compute_representative_scores(self, hidden_states: torch.Tensor, logits: torch.Tensor):
        """Compute scores for classes with representatives."""
        # Compute normal representative scores
        normal_scores = None
        if self.normal_representatives.shape[0] > 0:
            normal_scores = torch.matmul(hidden_states, self.normal_representatives.t())

        # Prepare MLP input if using MLP mode
        if self.use_mlp:
            mlp_input = torch.full_like(logits, self.empty_class_value)

            # Fill MLP input with scores
            if normal_scores is not None:
                for class_idx in self.classes_with_reps:
                    start_idx, end_idx = self.class_to_rep_indices[class_idx]
                    class_scores = normal_scores[:, start_idx:end_idx]
                    mlp_input[:, class_idx] = torch.max(class_scores, dim=1)[0]

            # Negative classes get 0.0 score
            for class_idx in self.negative_classes:
                mlp_input[:, class_idx] = 0.0

            # Apply MLP to get final logits
            logits.copy_(self.mlp(mlp_input))

        else:  # Max pooling mode
            # Fill logits directly with max scores
            if normal_scores is not None:
                for class_idx in self.classes_with_reps:
                    start_idx, end_idx = self.class_to_rep_indices[class_idx]
                    class_scores = normal_scores[:, start_idx:end_idx]
                    logits[:, class_idx] = torch.max(class_scores, dim=1)[0]

            # Negative classes get 0.0 score
            for class_idx in self.negative_classes:
                logits[:, class_idx] = 0.0


# Example usage:
