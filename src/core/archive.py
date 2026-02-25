"""
Archive implementation for MAP-Elites algorithm.
Stores the best prompt for each cell in the behavioral space grid.
"""

import numpy as np
import pickle
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict
import json


@dataclass
class ArchiveCell:
    """Represents a single cell in the MAP-Elites archive."""
    prompt: str
    behavior: Tuple[float, float]  # (a1, a2)
    quality: float  # Alignment Deviation score
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'prompt': self.prompt,
            'behavior': self.behavior,
            'quality': self.quality,
            'metadata': self.metadata
        }


class Archive:
    """
    MAP-Elites archive for storing behavioral diversity.
    
    The archive is a grid where each cell stores the best prompt
    that maps to that region of behavioral space.
    """
    
    def __init__(self, grid_size: int = 25):
        """
        Initialize the archive.
        
        Args:
            grid_size: Number of cells per dimension (total cells = grid_size^2)
        """
        self.grid_size = grid_size
        self.cells = np.empty((grid_size, grid_size), dtype=object)
        self.iteration_added = np.full((grid_size, grid_size), -1, dtype=int)
        self.current_iteration = 0
        
    def _behavior_to_cell(self, behavior: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert continuous behavior coordinates to discrete cell indices.
        
        Args:
            behavior: (a1, a2) in [0, 1]^2
            
        Returns:
            (i, j) cell indices
        """
        a1, a2 = behavior
        # Clip to [0, 1] to handle edge cases
        a1 = np.clip(a1, 0.0, 0.9999)
        a2 = np.clip(a2, 0.0, 0.9999)
        
        i = int(a1 * self.grid_size)
        j = int(a2 * self.grid_size)
        
        return i, j
    
    def add(self, prompt: str, behavior: Tuple[float, float], quality: float, 
            metadata: Optional[Dict] = None) -> bool:
        """
        Add a prompt to the archive if it's better than the existing one in its cell.
        
        Args:
            prompt: The prompt text
            behavior: (a1, a2) behavioral descriptor
            quality: Alignment Deviation score
            metadata: Optional metadata (iteration, timestamp, etc.)
            
        Returns:
            True if the prompt was added (new cell or better quality), False otherwise
        """
        i, j = self._behavior_to_cell(behavior)
        
        # Accept if: empty cell with quality > 0.05, or better than existing
        if self.cells[i, j] is None:
            if quality > 0.05:  # Lower threshold for exploration
                if metadata is None:
                    metadata = {}
                metadata['iteration'] = self.current_iteration
                
                self.cells[i, j] = ArchiveCell(
                    prompt=prompt,
                    behavior=behavior,
                    quality=quality,
                    metadata=metadata
                )
                self.iteration_added[i, j] = self.current_iteration
                return True
        elif quality > self.cells[i, j].quality:
            if metadata is None:
                metadata = {}
            metadata['iteration'] = self.current_iteration
            
            self.cells[i, j] = ArchiveCell(
                prompt=prompt,
                behavior=behavior,
                quality=quality,
                metadata=metadata
            )
            self.iteration_added[i, j] = self.current_iteration
            return True
        
        return False
    
    def get_cell(self, behavior: Tuple[float, float]) -> Optional[ArchiveCell]:
        """Get the cell at the given behavior coordinates."""
        i, j = self._behavior_to_cell(behavior)
        return self.cells[i, j]
    
    def select_random(self) -> Optional[ArchiveCell]:
        """
        Select a random non-empty cell from the archive.
        
        Returns:
            A random ArchiveCell, or None if archive is empty
        """
        non_empty = [(i, j) for i in range(self.grid_size) 
                     for j in range(self.grid_size) 
                     if self.cells[i, j] is not None]
        
        if not non_empty:
            return None
        
        i, j = non_empty[np.random.randint(len(non_empty))]
        return self.cells[i, j]
    
    def select_fitness_proportionate(self) -> Optional[ArchiveCell]:
        """
        Select a cell with probability proportional to its quality score.
        
        Returns:
            An ArchiveCell selected via roulette wheel selection
        """
        non_empty = [(i, j) for i in range(self.grid_size) 
                     for j in range(self.grid_size) 
                     if self.cells[i, j] is not None]
        
        if not non_empty:
            return None
        
        # Get qualities
        qualities = np.array([self.cells[i, j].quality for i, j in non_empty])
        
        # Add small epsilon to avoid zero probabilities
        qualities = qualities + 1e-6
        
        # Normalize to probabilities
        probs = qualities / qualities.sum()
        
        # Sample
        idx = np.random.choice(len(non_empty), p=probs)
        i, j = non_empty[idx]
        return self.cells[i, j]
    
    def get_coverage(self) -> float:
        """
        Calculate the percentage of filled cells.
        
        Returns:
            Coverage as a percentage (0-100)
        """
        filled = sum(1 for i in range(self.grid_size) 
                     for j in range(self.grid_size) 
                     if self.cells[i, j] is not None)
        total = self.grid_size * self.grid_size
        return 100.0 * filled / total
    
    def get_diversity(self, threshold: float = 0.5) -> int:
        """
        Count the number of cells with quality above threshold.
        
        Args:
            threshold: Minimum quality score to count as a failure mode
            
        Returns:
            Number of distinct failure modes
        """
        count = sum(1 for i in range(self.grid_size) 
                    for j in range(self.grid_size) 
                    if self.cells[i, j] is not None and 
                    self.cells[i, j].quality > threshold)
        return count
    
    def get_peak_quality(self) -> float:
        """
        Get the maximum quality score in the archive.
        
        Returns:
            Peak Alignment Deviation score
        """
        qualities = [self.cells[i, j].quality 
                     for i in range(self.grid_size) 
                     for j in range(self.grid_size) 
                     if self.cells[i, j] is not None]
        return max(qualities) if qualities else 0.0
    
    def get_qd_score(self) -> float:
        """
        Calculate the QD-score (sum of all quality scores).
        
        Returns:
            Sum of quality scores across all filled cells
        """
        return sum(self.cells[i, j].quality 
                   for i in range(self.grid_size) 
                   for j in range(self.grid_size) 
                   if self.cells[i, j] is not None)
    
    def to_heatmap(self) -> np.ndarray:
        """
        Convert archive to a 2D heatmap of quality scores.
        
        Returns:
            grid_size x grid_size array of quality scores (NaN for empty cells)
        """
        heatmap = np.full((self.grid_size, self.grid_size), np.nan)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.cells[i, j] is not None:
                    heatmap[i, j] = self.cells[i, j].quality
        return heatmap
    
    def get_all_prompts(self) -> List[Tuple[str, Tuple[float, float], float]]:
        """
        Get all prompts in the archive.
        
        Returns:
            List of (prompt, behavior, quality) tuples
        """
        prompts = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.cells[i, j] is not None:
                    cell = self.cells[i, j]
                    prompts.append((cell.prompt, cell.behavior, cell.quality))
        return prompts
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the archive.
        
        Returns:
            Dictionary of statistics
        """
        qualities = [self.cells[i, j].quality 
                     for i in range(self.grid_size) 
                     for j in range(self.grid_size) 
                     if self.cells[i, j] is not None]
        
        if not qualities:
            return {
                'coverage': 0.0,
                'diversity': 0,
                'peak_quality': 0.0,
                'mean_quality': 0.0,
                'std_quality': 0.0,
                'qd_score': 0.0,
                'num_filled': 0
            }
        
        return {
            'coverage': self.get_coverage(),
            'diversity': self.get_diversity(),
            'peak_quality': self.get_peak_quality(),
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'qd_score': self.get_qd_score(),
            'num_filled': len(qualities)
        }
    
    def save(self, filepath: str):
        """Save archive to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'grid_size': self.grid_size,
                'cells': self.cells,
                'iteration_added': self.iteration_added,
                'current_iteration': self.current_iteration
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Archive':
        """Load archive from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        archive = cls(grid_size=data['grid_size'])
        archive.cells = data['cells']
        archive.iteration_added = data['iteration_added']
        archive.current_iteration = data['current_iteration']
        return archive
    
    def export_to_json(self, filepath: str):
        """Export archive contents to JSON for analysis."""
        data = {
            'grid_size': self.grid_size,
            'current_iteration': self.current_iteration,
            'statistics': self.get_statistics(),
            'cells': []
        }
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.cells[i, j] is not None:
                    cell = self.cells[i, j]
                    data['cells'].append({
                        'grid_position': (i, j),
                        'prompt': cell.prompt,
                        'behavior': cell.behavior,
                        'quality': cell.quality,
                        'metadata': cell.metadata
                    })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def increment_iteration(self):
        """Increment the iteration counter."""
        self.current_iteration += 1
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"Archive(grid_size={self.grid_size}, "
                f"coverage={stats['coverage']:.1f}%, "
                f"diversity={stats['diversity']}, "
                f"peak_quality={stats['peak_quality']:.3f})")
