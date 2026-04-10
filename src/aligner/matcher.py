import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Union
from abc import ABC, abstractmethod
from scipy.special import logsumexp

class BaseMatcher(ABC):
    @abstractmethod
    def match(self, obs_coords: np.ndarray, ref_coords: np.ndarray, **kwargs):
        pass

class HungarianMatcher(BaseMatcher):
    """
    Solves the 1-to-1 correspondence problem between an embryo and a reference frame.
    Minimizes the sum of squared Euclidean distances between points. 
    Supports optional slack augmentation for unassigned cells.
    """
    
    def __init__(self, tau: float = 1e6):
        self.tau = tau
    
    def match(self, obs_coords: np.ndarray, ref_coords: np.ndarray, tau: float = None, 
              use_slack: bool = True, return_matrix: bool = True, **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        
        current_tau = tau if tau is not None else self.tau
        C = cdist(obs_coords, ref_coords, metric="sqeuclidean")
        N, M = C.shape
        
        if use_slack:
            total_size = N + M
            C_aug = np.full((total_size, total_size), current_tau)
            C_aug[:N, :M] = C
            C_aug[N:, M:] = 0  # Slack-to-slack matches are free
            
            row_ind_full, col_ind_full = linear_sum_assignment(C_aug)
            final_col_ind = col_ind_full[:N]
        else:
            # Strict 1-to-1 matching
            _, final_col_ind = linear_sum_assignment(C)
        
        if return_matrix:
            P = np.zeros((N, M))
            for i, target_idx in enumerate(final_col_ind):
                if target_idx < M:  # Only assign if it matched a real reference cell
                    P[i, target_idx] = 1.0
            return P
            
        return np.arange(N), final_col_ind
    
    
class SinkhornMatcher(BaseMatcher):
    """
    Solves the correspondence problem using entropic regularization. 
    Allows for 'unassigned' probability mass to handle trash cells or outliers.
    """
    def __init__(self, epsilon: float = 0.05, max_iters: int = 100, stop_thr: float = 1e-3):
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.stop_thr = stop_thr
        
    def match(self, obs_coords: np.ndarray, ref_coords: np.ndarray, tau: float = 1e6, 
              epsilon: float = None, use_slack: bool = True, return_matrix: bool = True, 
              **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        
        current_eps = epsilon if epsilon is not None else self.epsilon
        P = self.compute_P(obs_coords, ref_coords, tau=tau, epsilon=current_eps, use_slack=use_slack)
        
        if return_matrix:
            return P
            
        # Hardening: Return discrete assignments
        row_ind = np.arange(len(obs_coords))
        col_ind = np.argmax(P, axis=1)
        return row_ind, col_ind
    
    def compute_P(self, obs_coords: np.ndarray, ref_coords: np.ndarray, tau: float = 1e6, 
                  epsilon: float = None, use_slack: bool = True) -> np.ndarray:
        
        C = cdist(obs_coords, ref_coords, metric="sqeuclidean")
        N, M = C.shape
        eps = epsilon if epsilon is not None else self.epsilon
        
        if use_slack:
            C_aug = np.full((N + 1, M + 1), tau)
            C_aug[:N, :M] = C
            C_aug[N, M] = 0
        else:
            C_aug = C
        
        # Log-domain Sinkhorn using logsumexp for numerical stability
        log_K = -C_aug / eps
        f = np.zeros(C_aug.shape[0])
        g = np.zeros(C_aug.shape[1])
        
        for _ in range(self.max_iters):
            f_prev = f.copy()
            
            # Replaced manual log/sum/exp with mathematically stable logsumexp
            f = -eps * logsumexp((g[None, :] + log_K * eps) / eps, axis=1)
            g = -eps * logsumexp((f[:, None] + log_K * eps) / eps, axis=0)
            
            if np.linalg.norm(f - f_prev) < self.stop_thr:
                break
                
        P_full = np.exp((f[:, None] + g[None, :] + log_K * eps) / eps)
        
        return P_full[:N, :M]