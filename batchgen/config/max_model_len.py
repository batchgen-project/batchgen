"""Deployment sequence-length contract, independent of CUDA and model backends."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SequenceLengthCapacity:
    """One region's capacity for a single sequence, including all its layers."""

    resource: str
    region: str
    node: int
    rank: int
    page_size_tokens: int
    bytes_per_page: int
    available_pages: int
    copies: int = 1

    def required_pages(self, length: int) -> int:
        return ((length + self.page_size_tokens - 1) // self.page_size_tokens) * self.copies

    @property
    def feasible_length(self) -> int:
        return max(0, self.available_pages // self.copies) * self.page_size_tokens


class MaxModelLenCapacityError(ValueError):
    """A deployment cannot hold one sequence of its promised length."""

    def __init__(self, length: int, capacity: SequenceLengthCapacity):
        self.max_model_len = length
        self.resource = capacity.resource
        self.region = capacity.region
        self.node = capacity.node
        self.rank = capacity.rank
        self.required_pages = capacity.required_pages(length)
        self.available_pages = capacity.available_pages
        self.required_bytes = self.required_pages * capacity.bytes_per_page
        self.available_bytes = self.available_pages * capacity.bytes_per_page
        self.estimated_feasible_length = capacity.feasible_length
        super().__init__(
            f"max_model_len={length} cannot fit one sequence: {self.resource} "
            f"region={self.region} node={self.node} rank={self.rank} "
            f"requires {self.required_pages} pages ({self.required_bytes} bytes), "
            f"available {self.available_pages} pages ({self.available_bytes} bytes); "
            f"estimated feasible max_model_len={self.estimated_feasible_length}. "
            "Lower --max-model-len or opt in to --max-model-len=-1."
        )


def resolve_max_model_len(requested: int | None, model_context_length: int) -> int:
    """Resolve a launch request; only -1 opts into later capacity fitting."""
    if model_context_length < 2:
        raise ValueError("model context length must allow a prompt and a decode token")
    if requested is None or requested == -1:
        return model_context_length
    if requested < 2 or requested > model_context_length:
        raise ValueError(
            f"max_model_len must be between 2 and model context length "
            f"{model_context_length}, or -1 for auto-fit; got {requested}"
        )
    return requested


def verify_max_model_len(
    length: int, capacities: Sequence[SequenceLengthCapacity], *, auto_fit: bool = False,
) -> int:
    """Check every rank/node/region; auto-fit alone may lower the deployment bound."""
    if not capacities:
        raise ValueError("max_model_len verification requires measured capacities")
    limiting = min(capacities, key=lambda item: item.feasible_length)
    feasible = min(length, limiting.feasible_length)
    if feasible < length and (not auto_fit or feasible < 2):
        raise MaxModelLenCapacityError(length, limiting)
    return feasible
