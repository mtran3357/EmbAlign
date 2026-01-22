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
        
    def fit(source: np.ndarray, target: np.ndarray) -> None:
        """
        Calculates the optimal rotation R and translation t using the 
        Kabsch algorithm to map: source @ R + t ~= target.
        """
        assert source.shape == target.shape
        
        # 1. Center
        mu_src = source.mean(axis=0)
        mu_tgt = target.mean(axis=0)
        src_c = source - mu_src
        tgt_c = target - mu_tgt
        
        # 2. Computer Cov Mat and SVD
        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        
        # 3. Find Rotation
        R_col = Vt.T @ U.T
        
        # 4. Handle Reflections
        if np.linalg.det(R_col) < 0:
            Vt[-1, :] *= -1
            R_col = Vt.T @ U.T
            
        self.R = Vt.T @ U.T
        
        # 5. Find Translation
        self.t = mu_tgt - mu_src @ self.R
        
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
        R_col = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s**2 + 1e-12))
        return R_col.T
    
    @staticmethod
    def get_rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Generates a row-based rotation matrix for 'angle' about 'axis.'
        """
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        x, y, z = axis
        c, s = math.cos(angle), math.sin(angle)
        C = 1.0 - c
        R_col = np.array([
            [c + x*x*C,    x*y*C - z*s,  x*z*C + y*s],
            [y*x*C + z*s,  c + y*y*C,    y*z*C - x*s],
            [z*x*C - y*s,  z*y*C + x*s,  c + z*z*C]
        ])
        return R_col.T