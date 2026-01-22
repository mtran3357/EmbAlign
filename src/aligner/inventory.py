import pandas as pd
from typing import List, Dict, Tuple

class SliceAtlas:
    """
    Manages the database of valid cell-type combinations (slices).
    """
    
    def __init__(self, slice_csv_path: str):
        self.slices_by_n: Dict[int, List[Tuple[str, ...]]] = {}
        self._load_slices(slice_csv_path)
    
    def _load_slices(self, path: str) -> None:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            n = int(row['n_cells_frame'])
            cell_names = tuple(sorted(str(row["cell_names"]).split(";")))
            if n not in self.slices_by_n:
                self.slices_by_n[n] = []
            if cell_names not in self.slices_by_n[n]:
                self.slices_by_n[n].append(cell_names)
    
    def get_candidates(self, n_cells: int) -> List[Tuple[str, ...]]:
        """
        Returns all valid cell-name combinations for a given cell count.
        """
        return self.slices_by_n.get(n_cells, [])