import numpy as np
import math
from typing import Tuple

class RigidTransformer:
    """
    Handles rigid body transformations (rotation + translation) to align 
    point clouds, specifically implementing the Kabsch algorithm.
    """
    
    def __init__(self):
        self.R = np.eye(3)
        self.t = np.zeros((1,3))
        
    def fit_weighted(self, source: np.ndarray, target: np.ndarray, P: np.ndarray):
        """
        Optimal R and t for source @ R + t ~= (weights @ target)
        
        Args:
            source: (N, 3) Experimental normalized coordinates.
            target: (M, 3) Atlas reference means.
            weights: (N, M) assignment matrix (dense or binary).
        """
        # 1. Compute total mass and centroids
        # We use the row/column sums of P as weights for each point set
        w_source = np.sum(P, axis=1) # (N,)
        w_target = np.sum(P, axis=0) # (M,)
        total_mass = np.sum(w_source)
        
        if total_mass < 1e-9:
            self.R, self.t = np.eye(3), np.zeros((1, 3))
            return

        mu_src = (w_source @ source) / total_mass
        mu_tgt = (w_target @ target) / total_mass

        # 2. Center the point clouds
        src_c = source - mu_src
        tgt_c = target - mu_tgt

        # 3. Compute Weighted Covariance Matrix H
        # H = src_c^T @ P @ tgt_c
        H = src_c.T @ P @ tgt_c
        
        # 4. SVD to find optimal rotation
        U, S, Vt = np.linalg.svd(H)
        
        # Calculate R for column convention first
        R_candidate = Vt.T @ U.T
        
        # 5. Handle Reflection (Right-handed coordinate system)
        if np.linalg.det(R_candidate) < 0:
            Vt[-1, :] *= -1
            R_candidate = Vt.T @ U.T
            
        # 6. Convert to Row Convention for use in 'coords @ R'
        self.R = R_candidate.T 
        self.t = (mu_tgt - mu_src @ self.R).reshape(1, 3)
        
    def transform(self, coords: np.ndarray) -> np.ndarray:
        """
        Applies current R and to to the given coordinates.
        """
        return coords @ self.R + self.t
    
    @staticmethod
    def get_rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Creates a rotation matrix R such that a @ R aligns with vector b.
        Useful for aligning PC1 axes.
        """
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        
        v = np.cross(a_norm, b_norm)
        s = np.linalg.norm(v)
        c = float(np.dot(a_norm, b_norm))
        
        if s < 1e-8:
            return np.eye(3) if c > 0 else -np.eye(3)
        
        # Rodrigues' rotation formula
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s**2))
        return R.T
    
    @staticmethod
    def get_rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Generates a row-based rotation matrix for 'angle' about 'axis.'
        """
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        x, y, z = axis
        c, s = math.cos(angle), math.sin(angle)
        C = 1.0 - c
        return np.array([
            [c + x*x*C,    x*y*C + z*s,  x*z*C - y*s],
            [y*x*C - z*s,  c + y*y*C,    y*z*C + x*s],
            [z*x*C + y*s,  z*y*C - x*s,  c + z*z*C]
        ])