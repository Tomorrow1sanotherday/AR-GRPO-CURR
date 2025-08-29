import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import Iterator, Optional, List, Dict, Any, Sized
from collections import defaultdict
import json

class CurriculumRepeatSampler(Sampler):
    """
    Sampler that combines curriculum learning strategies with repeat sampling functionality.
    
    Supports four curriculum learning strategies:
    - timestep: Fixed order based on difficulty (easy to hard)
    - balance: Random uniform sampling across all difficulties
    - cosine: Smooth transition from easy to hard using cosine schedule
    - gaussian: Bell-curve distribution with adjustable parameters
    
    Also supports RepeatSampler functionality:
    - mini_repeat_count: Number of times to repeat each index per batch
    - batch_size: Number of unique indices per batch
    - repeat_count: Number of times to repeat the full sampling process
    """

    def __init__(
        self,
        data_source: Sized,
        # RepeatSampler parameters
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        # Curriculum learning parameters
        strategy: str = "balance",
        total_steps: int = 300,
        alpha: float = 1.0,  # Gaussian width parameter
        beta: float = 1.0,   # Gaussian progression speed parameter
    ):
        """
        Initialize curriculum repeat sampler.

        Args:
            data_source: Dataset to sample from
            mini_repeat_count: Number of times to repeat each index per batch
            batch_size: Number of unique indices per batch
            repeat_count: Number of times to repeat the full sampling process
            shuffle: Whether to shuffle the dataset (affects non-curriculum strategies)
            seed: Random seed for reproducibility
            strategy: Sampling strategy ("timestep", "balance", "cosine", "gaussian")
            total_steps: Total steps needed for cosine and gaussian sampler
            alpha: Gaussian schedule width parameter (larger = wider difficulty spread)
            beta: Gaussian schedule progression speed (larger = faster progression to hard samples)
        """
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        # Curriculum learning parameters
        if strategy not in ["timestep", "balance", "cosine", "gaussian"]:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of: timestep, balance, cosine, gaussian")
        
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta
        self.total_steps = total_steps
        self.current_epoch = 0

        # Build difficulty mapping from dataset
        self._build_difficulty_mapping()

        # Initialize random generator
        if shuffle or strategy in ["balance", "cosine", "gaussian"]:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def _build_difficulty_mapping(self) -> None:
        """Build mapping from difficulty levels to sample indices."""
        self.difficulty_to_indices: Dict[Any, List[int]] = defaultdict(list)

        for idx in range(len(self.data_source)):
            try:
                sample = self.data_source[idx]['prompt']
                if isinstance(sample, dict):
                    difficulty = sample.get("difficulty", 0)
                elif hasattr(sample, "difficulty"):
                    difficulty = sample.difficulty
                else:
                    difficulty = 0
            except:
                difficulty = 0
            self.difficulty_to_indices[difficulty].append(idx)
        
     
        # Sort difficulties and ensure at least one exists
        self.unique_difficulties = sorted(self.difficulty_to_indices.keys())
        self.num_difficulties = len(self.unique_difficulties)

        if self.num_difficulties == 0:
            self.unique_difficulties = [0]
            self.difficulty_to_indices[0] = list(range(len(self.data_source)))
            self.num_difficulties = 1

    def _get_timestep_indices(self) -> List[int]:
        """Return indices in fixed difficulty order (easy to hard)."""
        indices = []
        for difficulty in self.unique_difficulties:
            indices.extend(self.difficulty_to_indices[difficulty])
        return indices

    def _get_balance_indices(self) -> List[int]:
        """Return randomly shuffled indices (uniform sampling)."""
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=self.generator)
            return [indices[i] for i in perm]
        return indices

    def _get_cosine_indices(self) -> List[int]:
        """Return indices using cosine curriculum schedule."""
        # Calculate progress through training (0 to 1)
        progress = min(self.current_epoch / max(self.total_steps - 1, 1), 1.0)

        # Cosine weight: 1.0 (easy focus) at start, 0.0 (hard focus) at end
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Calculate sampling probabilities for each difficulty
        difficulty_probs = []
        for i in range(self.num_difficulties):
            # Normalize difficulty position to [0, 1]
            normalized_pos = i / max(self.num_difficulties - 1, 1)
            # Higher weight for easier difficulties early in training
            prob = cosine_weight * (1.0 - normalized_pos) + (1.0 - cosine_weight) * normalized_pos
            prob = max(prob, 0.05)  # Minimum probability to ensure some diversity
            difficulty_probs.append(prob)

        return self._sample_by_difficulty_probs(difficulty_probs)

    def _get_gaussian_indices(self) -> List[int]:
        """Return indices using Gaussian curriculum schedule."""
        # Calculate progress through training (0 to 1)
        progress = min(self.current_epoch / max(self.total_steps - 1, 1), 1.0)

        # Center of Gaussian moves from easy (0) to hard (num_difficulties-1)
        center = progress * (self.num_difficulties - 1) * self.beta

        # Calculate Gaussian probabilities for each difficulty
        difficulty_probs = []
        for i in range(self.num_difficulties):
            # Gaussian centered at 'center' with spread controlled by alpha
            prob = math.exp(-0.5 * ((i - center) / max(self.alpha, 0.1)) ** 2)
            difficulty_probs.append(prob)

        return self._sample_by_difficulty_probs(difficulty_probs)

    def _sample_by_difficulty_probs(self, difficulty_probs: List[float]) -> List[int]:
        """Sample dataset indices based on difficulty probabilities."""
        # Normalize probabilities
        total_prob = sum(difficulty_probs)
        if total_prob <= 0:
            difficulty_probs = [1.0] * len(difficulty_probs)
            total_prob = len(difficulty_probs)

        normalized_probs = [p / total_prob for p in difficulty_probs]

        # Calculate number of samples to draw from each difficulty
        samples_per_difficulty = []
        total_samples = len(self.data_source)

        for i, prob in enumerate(normalized_probs):
            if i == len(normalized_probs) - 1:
                # Last difficulty gets remaining samples
                remaining = total_samples - sum(samples_per_difficulty)
                samples_per_difficulty.append(max(0, remaining))
            else:
                num_samples = int(total_samples * prob)
                samples_per_difficulty.append(num_samples)

        # Sample indices from each difficulty level
        indices = []

        for difficulty, num_samples in zip(self.unique_difficulties, samples_per_difficulty):
            difficulty_indices = self.difficulty_to_indices[difficulty]

            if num_samples <= 0:
                continue
            elif num_samples >= len(difficulty_indices):
                # Use all indices from this difficulty (with repetition if needed)
                repeat_count = num_samples // len(difficulty_indices)
                remainder = num_samples % len(difficulty_indices)

                indices.extend(difficulty_indices * repeat_count)
                if remainder > 0:
                    perm = torch.randperm(len(difficulty_indices), generator=self.generator)[:remainder]
                    indices.extend([difficulty_indices[i] for i in perm])
            else:
                # Randomly sample from this difficulty
                perm = torch.randperm(len(difficulty_indices), generator=self.generator)[:num_samples]
                indices.extend([difficulty_indices[i] for i in perm])

        # Final shuffle of all selected indices
        if indices and self.shuffle:
            perm = torch.randperm(len(indices), generator=self.generator)
            indices = [indices[i] for i in perm]

        return indices

    def _get_curriculum_indices(self) -> List[int]:
        """Get indices based on curriculum learning strategy."""
        if self.strategy == "timestep":
            return self._get_timestep_indices()
        elif self.strategy == "balance":
            return self._get_balance_indices()
        elif self.strategy == "cosine":
            return self._get_cosine_indices()
        elif self.strategy == "gaussian":
            return self._get_gaussian_indices()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def __iter__(self):
        # Get indices based on curriculum learning strategy
        indexes = self._get_curriculum_indices()

        # Apply RepeatSampler logic
        # Split into batches
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        # Keep only full batches
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        # Apply repeat logic
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for curriculum learning progression."""
        self.current_epoch = epoch
        # Update random seed for this epoch
        if hasattr(self, 'generator') and self.seed is not None:
            self.generator.manual_seed(self.seed + epoch)

    def __len__(self) -> int:
        """Return the total number of samples considering repeat factors."""
        # Calculate number of full batches
        num_full_batches = len(self.data_source) // self.batch_size
        return num_full_batches * self.batch_size * self.mini_repeat_count * self.repeat_count