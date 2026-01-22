import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Tuple

class HungarianMatcher:
    """
    Solves the 1-to-1 correspondence problem between an embryo and a reference frame.
    
    This matches the legacy approach by minimizing the sum of squared Euclidean 
    distances between points.
    """
    
    def __init__(self):
        pass
    
    def match(self, obs_coords: np.ndarray, ref_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Compute Cost Matrix
        cost_matrix = cdist(obs_coords, ref_coords, metric = "sqeuclidean")
        
        # 2. Solve linear sum assignment (Hungarian). Gives pairing that minimizes total cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return row_ind, col_ind
    
    
