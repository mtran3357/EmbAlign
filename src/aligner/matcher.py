import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Tuple

class HungarianMatcher:
    """
    Solves the 1-to-1 correspondence problem between an embryo and a reference frame.
    
    This matches the legacy approach by minimizing the sum of squared Euclidean 
    distances between points. Updated to handle slack in final assignment.
    """
    
    def __init__(self, tau: float = 1.0):
        self.tau = tau
    
    def match(self, obs_coords: np.ndarray, ref_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Compute Base Cost Matrix
        C = cdist(obs_coords, ref_coords, metric="sqeuclidean")
        N, M = C.shape
        
        # 2. Augment Matrix with Slack
        # We create a square-ish augmentation to ensure rejection is an option for all points.
        total_size = N + M
        C_aug = np.full((total_size, total_size), self.tau)
        C_aug[:N, :M] = C
        C_aug[N:, M:] = 0  # Slack-to-slack matches are free.
        
        # 3. Solve augmented assignment
        row_ind_full, col_ind_full = linear_sum_assignment(C_aug)
        
        # 4. Return assignments for the N real observations only
        # Indices >= M represent cells matched to the 'Slack' bin.
        return np.arange(N), col_ind_full[:N]
    
    
class SinkhornMatcher:
    """
    Solves the correspondence problem using entropic regularization. Sinkhorn implementation with added slack row and column.
    Allows for 'unassigned' probability mass to handle trash cells or outliers.
    """
    def __init__(self, epsilon: float = 0.05, max_iters: int = 100, stop_thr: float = 1e-3):
        """
        Args:
            epsilon (float): Regularization strength. Lower values approach 
                            Hungarian-like 'hard' matching.
            max_iters (int): Maximum number of Sinkhorn iterations.
            stop_thr (float): Convergence threshold for row/column sums.
        """
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.stop_thr = stop_thr
        
    def match(self, obs_coords: np.ndarray, ref_coords: np.ndarray, tau: float = 1.0,
              return_matrix: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Versatile interface to support both Legacy and Soft-Alignment pipelines.
        Args:
            obs_coords: (N, 3) Experimental coordinates.
            ref_coords: (M, 3) Atlas means.
            return_matrix: If True, returns (N, M) P matrix. Else returns indices.
        """
        P = self.compute_P(obs_coords, ref_coords, tau=tau)
        
        if return_matrix:
            return P
            
        # Hardening: Return discrete assignments for LegacyEngine logic
        row_ind = np.arange(len(obs_coords))
        col_ind = np.argmax(P, axis=1)
        
        return row_ind, col_ind
    
    def compute_P(self, obs_coords: np.ndarray, ref_coords: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """
        Computes the (N, M) probability matrix using an internal (N+1, M+1) slack construction.
        
        Args:
            tau: The 'rejection' cost. Points further than sqrt(tau) from 
                 the atlas will tend to move their mass into the slack bin.
        """
        # Base Cost Matrix
        C = cdist(obs_coords, ref_coords, metric="sqeuclidean")
        N, M = C.shape
        
        # Augment matrix with slack
        C_aug = np.full((N + 1, M + 1), tau)
        C_aug[:N, :M] = C
        C_aug[N,M] = 0
        
        # Vectorized Sinkhorn on augmented matrix
        K = np.exp(-C / self.epsilon)
        u = np.ones(K.shape[N + 1])
        v = np.ones(K.shape[M + 1])
        
        for i in range(self.max_iters):
            u_prev = u.copy()
            # Update scaling vectors
            v = 1.0 / (K.T @ u + 1e-12)
            u = 1.0 / (K @ v + 1e-12)
            
            # Early exit if scaling vectors converge
            if i % 5 == 0 and np.linalg.norm(u - u_prev) < self.stop_thr:
                break
        
        # Construct full P and slice back to original dimensions
        P_full - u[:, None] * K * v[None, :]
        # Only return correspondence betwee nreal observations and real atlas points
        return P_full[:N, :M]
    

            