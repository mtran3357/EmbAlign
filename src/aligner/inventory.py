import pandas as pd
from typing import List, Dict, Tuple

class SliceAtlas:
    """
    Manages the database of valid cell-type combinations (slices).
    """
    
    def __init__(self, slice_csv_path: str):
            self.n_to_ids: Dict[int, List[int]] = {}
            self.id_to_labels: Dict[int, Tuple[str, ...]] = {}
            self._load_slices(slice_csv_path)
    
    def _load_slices(self, path: str) -> None:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            s_id = int(row['slice_id'])
            n = int(row['n_cells_frame'])
            labels = tuple(sorted(str(row["cell_names"]).split(";")))
            if n not in self.n_to_ids:
                self.n_to_ids[n] = []
            self.n_to_ids[n].append(s_id)
            self.id_to_labels[s_id] = labels
    
    def get_candidates(self, n_cells: int) -> List[int]:
        """Returns a list of slice_ids matching the cell count."""
        return self.n_to_ids.get(n_cells, [])

    def get_labels(self, slice_id: int) -> Tuple[str, ...]:
        """Retrieves the biological labels for a specific slice ID."""
        return self.id_to_labels.get(slice_id, ())