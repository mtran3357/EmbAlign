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

    def align_frame(self, frame, trace=False):
        """Standardizes and aligns an experimental frame against biological hypotheses."""
        frame.prepare()
        
        # Candidate selection
        candidate_ids = self.slice_db.get_candidates(len(frame))
        best_overall_result = None
        
        # Trace to map loss landscape
        landscape_traces = {}
        
        tau = self.settings.get('tau', 1.0)
        for s_id in candidate_ids:
            # Build slice models
            labels = self.slice_db.get_labels(s_id)
            ref_frame = ReferenceFrame(labels, self.atlas)
            
            # Coarse scan
            best_R_init, _, coarse_history = self._run_coarse_scan(frame, ref_frame, return_trace=trace)
            
            # ICP Refinement
            refined_R, refined_t, icp_history = self._refine_icp(frame, ref_frame, best_R_init, return_trace=trace)
            
            # Label and score
            aligned_coords = frame.normalized_coords @ refined_R + refined_t
            final_cost, assignments = self._final_mah_score(aligned_coords, ref_frame, tau=tau)
            # Map indices to labels, handling the 'Slack' rejection index
            # If an assignment index is >= n_real, it was matched to the slack bin
            final_labels = []
            for idx in assignments:
                if idx < ref_frame.n_real:
                    final_labels.append(ref_frame.labels[idx])
                else:
                    final_labels.append("unassigned")
            # Log trace if requested
            if trace:
                landscape_traces[s_id] = {
                    'coarse': coarse_history,
                    'icp_history': icp_history,
                    'final_mahalanobis': final_cost
                }
            
            # Track winner
            if best_overall_result is None or final_cost < best_overall_result['cost']:
                best_overall_result = {
                    'slice_id': s_id,
                    'cost': final_cost,
                    'labels': final_labels,
                    'coords': aligned_coords,
                    'scale_factor': frame.scale_factor
                }
            
        if trace:
                return best_overall_result, landscape_traces
            
        return best_overall_result
        
    def _run_coarse_scan(self, frame, ref_frame, return_trace=False):
        """PC1 based rotation scan."""
        best_cost = float('inf')
        best_R = None
        
        # Trace storage
        trace_data = [] if return_trace else None
        
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

                # Record trace if requested
                if return_trace:
                    trace_data.append({
                        'sign': sign,
                        'angle_deg': np.degrees(i * self.angle_step_rad),
                        'cost': cost
                    })
                
                if cost < best_cost:
                    best_cost = cost
                    best_R = R_total
                        
        return best_R, best_cost, trace_data
    
    def _refine_icp(self, frame, ref_frame, initial_R, return_trace=False):
        """Snap alignment into place with Euclidean ICP."""
        R_curr = initial_R
        t_curr = ref_frame.center_of_mass
        
        # Trace
        trace_data =[] if return_trace else None
        
        for i in range(self.settings.get('icp_iters', 5)):
            current_pts = frame.normalized_coords @ R_curr + t_curr        
            # Hybrid Assignment
            if hasattr(self.matcher, 'compute_P'):
                # Sinkhorn Path
                W = self.matcher.match(current_pts, ref_frame.means, return_matrix = True)
            else:
                # Hungarian path
                _, col_ind = self.matcher.match(current_pts, ref_frame.means)
                W = np.zeros((len(current_pts), len(ref_frame.means)))
                W[np.arange(len(current_pts)), col_ind] = 1.0
            # Trace optionally
            if return_trace:
                # Calculate Euclidean cost for this iteration
                diff = current_pts - ref_frame.means[col_ind]
                cost = np.sum(diff**2)
                trace_data.append({'iteration': i, 'cost': cost})
            # Calculate rigid transform for current correspondences
            self.transformer.fit_weighted(frame.normalized_coords, ref_frame.means, W)
            R_curr, t_curr = self.transformer.R, self.transformer.t
            
        return R_curr, t_curr, trace_data
    
    def _final_mah_score(self, aligned_coords, ref_frame, tau=1.0):
        """Vectorized Mahalanobis scoring for final label assignment."""
        N = len(aligned_coords)
        M = ref_frame.n_real
        
        D = np.zeros((N, M))
        for i in range(N):
            mu = ref_frame.means[i]
            inv_cov = ref_frame.inv_covs[i]
            diff = aligned_coords - mu
            # Mahalanobis: (x-mu)^T * InvCov * (x-mu)
            D[:, i] = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        
        total_size = N + M
        D_aug = np.full((total_size, total_size), tau)
        D_aug[:N, :M] = D
        D_aug[N:, M:] = 0
        
        row_ind_full, col_ind_full = linear_sum_assignment(D_aug)
        
        # We only care about the assignments for our N real observations
        final_assignments = col_ind_full[:N]
        total_cost = D_aug[np.arange(N), final_assignments].sum()
        
        return total_cost, final_assignments