from typing import List, Dict, Union, Optional, Tuple, Any, Set
from collections import defaultdict
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.utils import logging
from tokenizers import AddedToken, Tokenizer
import torch
import json
import os


class ScaffoldTokenizer(PreTrainedTokenizerFast):
    """
    A tokenizer that can replace scaffold tokens with their decompositions.

    This implementation works properly with transformers' loading mechanism
    by following the expected patterns for custom tokenizer classes.
    """

    def __init__(
        self,
        tokenizer_object=None,
        tokenizer_file=None,
        scaffold_vocab: Dict[str, int] = None,
        # For backward compatibility with your existing code
        extended_tokenizer: Optional[PreTrainedTokenizerFast] = None,
        base_tokenizer: Optional[PreTrainedTokenizerFast] = None,
        _from_pretrained: bool = False,
        **kwargs,
    ):
        """
        Initialize the ScaffoldTokenizer.

        Args:
            tokenizer_object: A tokenizers.Tokenizer object
            tokenizer_file: Path to a tokenizer.json file
            scaffold_vocab: Dictionary mapping scaffold tokens to their IDs
            extended_tokenizer: For backward compatibility - existing tokenizer to extend
            base_tokenizer: For backward compatibility - base tokenizer
            _from_pretrained: For backward compatibility - loading from pretrained
            **kwargs: Additional arguments passed to PreTrainedTokenizerFast
        """

        # Handle backward compatibility with your existing constructor pattern
        if extended_tokenizer is not None:
            # Your original pattern - use the tokenizer_object from extended_tokenizer
            tokenizer_object = extended_tokenizer.backend_tokenizer

            # Copy attributes from extended_tokenizer if needed
            if hasattr(extended_tokenizer, "added_tokens_decoder") and base_tokenizer is not None:
                kwargs.setdefault(
                    "added_tokens_decoder", base_tokenizer.added_tokens_decoder.copy()
                )

        # Make sure we have a tokenizer_object (required for PreTrainedTokenizerFast)
        if tokenizer_object is None and tokenizer_file is None:
            raise ValueError("Either tokenizer_object or tokenizer_file must be provided")

        # Initialize the parent class
        super().__init__(tokenizer_object=tokenizer_object, tokenizer_file=tokenizer_file, **kwargs)

        # Copy additional attributes from extended_tokenizer if provided
        if extended_tokenizer is not None:
            for attr, value in extended_tokenizer.__dict__.items():
                if not attr.startswith("_") and not hasattr(self, attr):
                    setattr(self, attr, value)

        # Initialize scaffold-specific attributes
        self.scaffold_vocab = scaffold_vocab or {}
        self.scaffold_id_map: Dict[int, List[int]] = {}
        self.scaffold_token_ids: Set[int] = set()

        # Build scaffold mappings
        self._build_scaffold_mappings()

    def _build_scaffold_mappings(self):
        """Build efficient mappings for scaffold token IDs to their shortest non-scaffold decomposition."""
        if not hasattr(self, "get_vocab") or not callable(getattr(self, "get_vocab")):
            # Skip if vocab isn't available yet (during initialization)
            return

        vocab = self.get_vocab()

        # Clear existing mappings
        self.scaffold_id_map.clear()
        self.scaffold_token_ids.clear()

        # Helper function to get all possible tokenizations
        def get_tokenizations(text: str) -> List[List[int]]:
            n = len(text)
            dp = [[] for _ in range(n + 1)]  # dp[i] stores all valid tokenizations up to position i
            dp[0] = [[]]  # empty sequence for empty string

            for i in range(1, n + 1):
                for j in range(i):
                    substr = text[j:i]
                    if substr in vocab:
                        token_id = vocab[substr]
                        # Only consider this tokenization if the token is not a scaffold
                        # (unless it's a single character token that we can't break down further)
                        if (substr not in self.scaffold_vocab) or (len(substr) == 1):
                            for prev_tokens in dp[j]:
                                dp[i].append(prev_tokens + [token_id])

            return dp[n]

        # Build mappings for each scaffold token
        for scaffold, scaffold_id in self.scaffold_vocab.items():
            if scaffold in vocab:
                self.scaffold_token_ids.add(scaffold_id)

                # Get all possible non-scaffold tokenizations
                tokenizations = get_tokenizations(scaffold)

                if tokenizations:
                    # Choose the shortest valid tokenization
                    shortest = min(tokenizations, key=len)
                    if shortest:  # Only store if we found a valid decomposition
                        self.scaffold_id_map[scaffold_id] = shortest
                else:
                    # If no valid tokenization found (e.g., for single character scaffolds),
                    # we keep the original token
                    self.scaffold_id_map[scaffold_id] = [scaffold_id]

    def _replace_scaffold_tokens(self, token_ids: List[int]) -> List[int]:
        """Replace scaffold tokens using pre-computed mapping."""
        result = []
        for token_id in token_ids:
            if (
                hasattr(self, "special_tokens_encoding_map")
                and token_id in self.special_tokens_encoding_map
            ):
                result.append(self.special_tokens_encoding_map[token_id])
            elif token_id in self.scaffold_token_ids:
                result.extend(self.scaffold_id_map[token_id])
            else:
                result.append(token_id)
        return result

    def _convert_tokens_to_ids(self, tokens):
        """Override to apply scaffold token replacement."""
        base_ids = super()._convert_tokens_to_ids(tokens)
        return self._replace_scaffold_tokens(base_ids)

    def to_dict(self):
        """Serialize the tokenizer to a dictionary."""
        # Get the parent class serialization
        serialized_data = super().to_dict()

        # Add scaffold-specific attributes
        serialized_data.update(
            {
                "scaffold_vocab": self.scaffold_vocab,
                "scaffold_token_ids": list(self.scaffold_token_ids),
                "scaffold_id_map": {str(k): v for k, v in self.scaffold_id_map.items()},
                "tokenizer_class": "ScaffoldTokenizer",  # This is expected by transformers
            }
        )

        return serialized_data

    def save_pretrained(self, save_directory, legacy_format=None, filename_prefix=None, **kwargs):
        """Save the tokenizer and its configuration file to a directory."""

        # Call the parent implementation
        result = super().save_pretrained(
            save_directory=save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            **kwargs,
        )

        return result

    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        use_auth_token=None,
        cache_dir=None,
        local_files_only=False,
        _commit_hash=None,
        _is_local=False,
        trust_remote_code=False,
        **kwargs,
    ):
        """
        This is the method that transformers calls during auto-loading.
        By implementing this, we handle the class resolution properly.
        """

        # Extract scaffold-specific configuration
        scaffold_vocab = init_configuration.pop("scaffold_vocab", {})
        scaffold_token_ids = set(init_configuration.pop("scaffold_token_ids", []))
        scaffold_id_map = {
            int(k): v for k, v in init_configuration.pop("scaffold_id_map", {}).items()
        }

        # Call the parent _from_pretrained to handle the standard loading
        tokenizer = super()._from_pretrained(
            resolved_vocab_files=resolved_vocab_files,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            init_configuration=init_configuration,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=_commit_hash,
            _is_local=_is_local,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # Set scaffold-specific attributes
        tokenizer.scaffold_vocab = scaffold_vocab
        tokenizer.scaffold_token_ids = scaffold_token_ids
        tokenizer.scaffold_id_map = scaffold_id_map

        # Rebuild mappings
        tokenizer._build_scaffold_mappings()

        return tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        """
        Load a tokenizer from a directory or the Hugging Face Hub.

        This delegates to the transformers auto-loading mechanism properly.
        """
        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

    # Alternative constructor for manual creation (your original use case)
    @classmethod
    def create_from_tokenizers(
        cls,
        extended_tokenizer: PreTrainedTokenizerFast,
        base_tokenizer: PreTrainedTokenizerFast,
        scaffold_vocab: Dict[str, int],
        **kwargs,
    ):
        """
        Create a ScaffoldTokenizer from existing tokenizers.

        This is for your original use case where you build the tokenizer manually.
        """
        # Use the extended_tokenizer pattern directly
        return cls(
            extended_tokenizer=extended_tokenizer,
            base_tokenizer=base_tokenizer,
            scaffold_vocab=scaffold_vocab,
            **kwargs,
        )

    # Keep other methods for backward compatibility
    def to_json_string(self):
        """Serialize the tokenizer to a JSON string."""
        serialized_data = self.to_dict()

        def convert_for_json(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        clean_data = convert_for_json(serialized_data)
        return json.dumps(clean_data, indent=2, sort_keys=True) + "\n"


# Register the tokenizer with transformers (this is the proper way)
try:
    from transformers import TOKENIZER_MAPPING_NAMES

    TOKENIZER_MAPPING_NAMES["ScaffoldTokenizer"] = "ScaffoldTokenizer"
except ImportError:
    # Fallback for older versions
    pass

try:
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    # Register our tokenizer in the mapping
    TOKENIZER_MAPPING.register("ScaffoldTokenizer", ScaffoldTokenizer)

except ImportError:
    pass
