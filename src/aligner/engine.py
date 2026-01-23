import numpy as np
from scipy.optimize import linear_sum_assignment
from aligner.models import ReferenceFrame
class LegacyEngine:
    def __init__(self, atlas, slice_db, matcher, transformer, settings: dict = None):
        self.atlas = atlas
        self.slice_db = slice_db
        self.matcher = matcher
        self.transformer = transformer
        self.settings = settings if settings is not None else {
            'angle_step_deg': 4.0,
            'icp_iters': 5
        }
        self.angle_step_rad = np.radians(self.settings['angle_step_deg'])

    def align_frame(self, frame):
        """Standardizes and aligns an experimental frame against biological hypotheses."""
        frame.prepare()
        
        # Candidate selection
        candidate_ids = self.slice_db.get_candidates(len(frame))
        best_overall_result = None
        
        for s_id in candidate_ids:
            # Build slice models
            labels = self.slice_db.get_labels(s_id)
            ref_frame = ReferenceFrame(labels, self.atlas)
            
            # Coarse scan
            best_R_init, _ = self._run_coarse_scan(frame, ref_frame)
            
            # ICP Refinement
            refined_R, refined_t = self._refine_icp(frame, ref_frame, best_R_init)
            
            # Label and score
            aligned_coords = frame.normalized_coords @ refined_R + refined_t
            final_cost, assignments = self._final_mah_score(aligned_coords, ref_frame)
            
            # Track winner
            if best_overall_result is None or final_cost < best_overall_result['cost']:
                best_overall_result = {
                    'slice_id': s_id,
                    'cost': final_cost,
                    'labels': [ref_frame.labels[i] for i in assignments],
                    'coords': aligned_coords,
                    'scale_factor': frame.scale_factor
                }
            
        return best_overall_result
        
    def _run_coarse_scan(self, frame, ref_frame):
        """PC1 based rotation scan."""
        best_cost = float('inf')
        best_R = None
        
        target_axis = ref_frame.pc1_axis
        # Scan both PC1 orientations
        for sign in [+1.0, -1.0]:
            R_initial = self.transformer.get_rotation_between_vectors(
                sign * frame.pc1_axis, target_axis
            )
            
            n_steps = int(2 * np.pi / self.angle_step_rad)
            for i in range(n_steps):
                R_roll = self.transformer.get_rotation_about_axis(
                    target_axis, i * self.angle_step_rad
                )
                R_total = R_initial @ R_roll
                
                # Center 
                transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
                _, col_ind = self.matcher.match(transformed, ref_frame.means)
                
                # Cost must be calculated in absolute atlas space
                diff = transformed - ref_frame.means[col_ind]
                cost = np.sum(diff**2) 
            
                if cost < best_cost:
                    best_cost = cost
                    best_R = R_total
                        
        return best_R, best_cost
    
    def _refine_icp(self, frame, ref_frame, initial_R):
        """Snap alignment into place with Euclidean ICP."""
        R_curr = initial_R
        t_curr = ref_frame.center_of_mass
        
        for _ in range(self.settings.get('icp_iters', 5)):
            current_pts = frame.normalized_coords @ R_curr + t_curr
            _, col_ind = self.matcher.match(current_pts, ref_frame.means)
            
            # Calculate rigid transform for current correspondences
            self.transformer.fit(frame.normalized_coords, ref_frame.means[col_ind])
            R_curr, t_curr = self.transformer.R, self.transformer.t
            
        return R_curr, t_curr
    
    def _final_mah_score(self, aligned_coords, ref_frame):
        """Vectorized Mahalanobis scoring for final label assignment."""
        N = len(aligned_coords)
        D = np.zeros((N,N))
        
        for i in range(N):
            mu = ref_frame.means[i]
            inv_cov = ref_frame.inv_covs[i]
            diff = aligned_coords - mu
            # Mahalanobis: (x-mu)^T * InvCov * (x-mu)
            D[:, i] = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        
        row_ind, col_ind = linear_sum_assignment(D)                    
        return D[row_ind, col_ind].sum(), col_ind