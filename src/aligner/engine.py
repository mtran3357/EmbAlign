import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from aligner.models import ReferenceFrame
from aligner.matcher import HungarianMatcher, SinkhornMatcher

class BaseEngine:
    def __init__(self, atlas, slice_db, matcher, transformer, settings=None):
        self.atlas = atlas
        self.slice_db = slice_db
        self.matcher = matcher
        self.transformer = transformer
        self.settings = settings or {}
        self.angle_step_rad = np.radians(self.settings.get('angle_step_deg', 4.0))
        
    def _final_mah_score(self, aligned_coords, ref_frame, tau=1e6):
        """Standard Mahalanobis scoring for the final 'Decision' phase."""
        N, M = len(aligned_coords), ref_frame.n_real
        D = np.zeros((N, M))
        for j in range(M):
            diff = aligned_coords - ref_frame.means[j]
            D[:, j] = np.einsum('ij,jk,ik->i', diff, ref_frame.inv_covs[j], diff)
        
        # Slack-based assignment
        C_aug = np.full((N + M, N + M), tau)
        C_aug[:N, :M] = D
        C_aug[N:, M:] = 0
        row, col = linear_sum_assignment(C_aug)
        final_col = col[:N]
        return C_aug[np.arange(N), final_col].sum(), final_col

class LegacyEngine(BaseEngine):
    """
    LegacyEngine: Implements the original rigid-body alignment algorithm.
    1. Discrete Hungarian matching (1-to-1).
    2. Single-start refinement (Winner-take-all from coarse scan).
    3. Euclidean-only cost minimization.
    """
    
    def align_frame(self, frame, trace=False):
        """Standardizes and aligns an experimental frame against biological hypotheses."""
        frame.prepare()
        candidate_ids = self.slice_db.get_candidates(len(frame))
        
        best_overall_result = None
        landscape_traces = {}
        
        # Use a very high tau for the coarse/refine phases to maintain 1-to-1 parity
        # The actual 'rejection' happens only in the final Mahalanobis scoring
        tau_strict = self.settings.get('tau_strict', 1e6)

        for s_id in candidate_ids:
            # Build biological reference hypothesis
            labels = self.slice_db.get_labels(s_id)
            ref_frame = ReferenceFrame(labels, self.atlas)
            
            # 1. PHASE A: Hard Coarse Scan (Single Winner)
            best_R_init, coarse_history = self._run_hard_scan(frame, ref_frame, return_trace=trace)
            
            # 2. PHASE B: Hard ICP Refinement
            refined_R, refined_t, icp_history = self._refine_hard_icp(
                frame, ref_frame, best_R_init, return_trace=trace
            )
            
            # 3. PHASE C: Biological Scoring (Inherited from BaseEngine)
            aligned_coords = frame.normalized_coords @ refined_R + refined_t
            final_cost, assignments = self._final_mah_score(
                aligned_coords, ref_frame, tau=self.settings.get('tau', 1e6)
            )
            
            # Map labels
            final_labels = [
                ref_frame.labels[idx] if idx < ref_frame.n_real else "unassigned" 
                for idx in assignments
            ]

            if trace:
                landscape_traces[s_id] = {
                    'coarse': coarse_history,
                    'icp_history': icp_history,
                    'final_mahalanobis': final_cost
                }

            # Track winner based on final Mahalanobis fit
            if best_overall_result is None or final_cost < best_overall_result['cost']:
                best_overall_result = {
                    'slice_id': s_id,
                    'cost': final_cost,
                    'labels': final_labels,
                    'coords': aligned_coords,
                    'init_angle': np.degrees(0) # Logic for legacy is single-start
                }
            
        if trace:
            return best_overall_result, landscape_traces
            
        return best_overall_result

    def _run_hard_scan(self, frame, ref_frame, return_trace=False):
        """PC1-based rotation scan using strict 1-to-1 Hungarian matching."""
        best_cost = float('inf')
        best_R = None
        trace_data = [] if return_trace else None
        
        target_axis = ref_frame.pc1_axis
        for sign in [+1.0, -1.0]:
            # Initial alignment of PC1 axes
            R_initial = self.transformer.get_rotation_between_vectors(
                sign * frame.pc1_axis, target_axis
            )
            
            n_steps = int(2 * np.pi / self.angle_step_rad)
            for i in range(n_steps):
                R_roll = self.transformer.get_rotation_about_axis(
                    target_axis, i * self.angle_step_rad
                )
                R_total = R_initial @ R_roll
                
                transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
                
                # Match returns (row_ind, col_ind) indices
                _, col_ind = self.matcher.match(transformed, ref_frame.means, tau=1e6)
                
                # Hard Euclidean Cost calculation
                diff = transformed - ref_frame.means[col_ind]
                cost = np.sum(diff**2) 

                if return_trace:
                    trace_data.append({
                        'sign': sign,
                        'angle_deg': np.degrees(i * self.angle_step_rad),
                        'cost': cost
                    })
                
                if cost < best_cost:
                    best_cost, best_R = cost, R_total
                        
        return best_R, trace_data

    def _refine_hard_icp(self, frame, ref_frame, initial_R, return_trace=False):
        """Discrete ICP using 1-to-1 correspondences."""
        R_curr, t_curr = initial_R, ref_frame.center_of_mass
        icp_iters = self.settings.get('icp_iters', 5)
        trace_data = [] if return_trace else None
        
        for i in range(icp_iters):
            current_pts = frame.normalized_coords @ R_curr + t_curr        
            
            # Get hard assignments
            _, col_ind = self.matcher.match(current_pts, ref_frame.means, tau=1e6)
            
            if return_trace:
                cost = np.sum((current_pts - ref_frame.means[col_ind])**2)
                trace_data.append({'iteration': i, 'cost': cost})
            
            # Create binary P matrix for the modular transformer
            P = np.zeros((len(current_pts), len(ref_frame.means)))
            P[np.arange(len(current_pts)), col_ind] = 1.0
            
            # Update transformation
            self.transformer.fit_weighted(frame.normalized_coords, ref_frame.means, P)
            R_curr, t_curr = self.transformer.R, self.transformer.t
            
        return R_curr, t_curr, trace_data

class EngineV1(BaseEngine):
    """
    EngineV1: A modular orchestrator implementing:
    1. Sinkhorn-based coarse scans for smooth landscapes.
    2. Multi-start 'Tournament' to escape geometric symmetry (120/180 degree traps).
    3. Weighted Kabsch ICP for soft-correspondence refinement.
    """
    
    def __init__(self, atlas, slice_db, matcher, transformer, settings: dict = None):
        self.atlas = atlas
        self.slice_db = slice_db
        self.matcher = matcher
        self.transformer = transformer
        self.settings = settings if settings is not None else {
            'angle_step_deg': 4.0,
            'icp_iters': 10,
            'tau': 1e6,
            'epsilon_coarse': 0.1,
            'epsilon_refine': 0.01,
            'k_tournament': 3
        }
        self.angle_step_rad = np.radians(self.settings.get('angle_step_deg', 4.0))

    def align_frame(self, frame, trace=False):
        """Coordinates the multi-start tournament across biological hypotheses."""
        frame.prepare()
        n_cells = len(frame)
        if n_cells < 10:
            k = self.settings.get('k_tournament', 3)
        else:
            k = 1
        
        candidate_ids = self.slice_db.get_candidates(len(frame))
        best_overall_result = None
        landscape_traces = {}

        tau = self.settings.get('tau', 1e6)
        for s_id in candidate_ids:
            # Build biological reference hypothesis
            labels = self.slice_db.get_labels(s_id)
            ref_frame = ReferenceFrame(labels, self.atlas)
            # Returns top unique starting orientations based on Euclidean/Sinkhorn cost
            valleys, coarse_history = self._find_top_k_valleys(frame, ref_frame, k=k, return_trace=trace)
            
            tournament_outcomes = []

            # Multi-Start Tournament (Refinement)
            for i, init in enumerate(valleys):
                # Run Soft-ICP refinement for this specific valley
                refined_R, refined_t, icp_history = self._refine_soft_icp(
                    frame, ref_frame, init['R'], return_trace=trace
                )
                
                # Biological Scoring (Mahalanobis)
                aligned_coords = frame.normalized_coords @ refined_R + refined_t
                final_cost, assignments = self._final_mah_score(aligned_coords, ref_frame, tau=tau)
                
                # Map labels with slack protection
                final_labels = [
                    ref_frame.labels[idx] if idx < ref_frame.n_real else "unassigned" 
                    for idx in assignments
                ]

                outcome = {
                    'slice_id': s_id,
                    'cost': final_cost,
                    'labels': final_labels,
                    'coords': aligned_coords,
                    'start_rank': i + 1,
                    'init_angle': init['angle'],
                    'init_sign': init['sign'],
                    'icp_trace': icp_history
                }
                tournament_outcomes.append(outcome)

                # Track the global biological winner
                if best_overall_result is None or final_cost < best_overall_result['cost']:
                    best_overall_result = outcome

            # Log trace data for the slice if requested
            if trace:
                landscape_traces[s_id] = {
                    'coarse': coarse_history,
                    'tournament': tournament_outcomes,
                    'ref_frame': ref_frame
                }
            
        if trace:
            return best_overall_result, landscape_traces
            
        return best_overall_result

    def _find_top_k_valleys(self, frame, ref_frame, k=3, return_trace=False):
        """Identifies unique basins in the rotation landscape using circular distance."""
        history = []
        target_axis = ref_frame.pc1_axis
        eps = self.settings.get('epsilon_coarse', 0.1)
        tau = self.settings.get('tau', 1e6) # High tau for coarse to prevent mass loss

        for sign in [+1.0, -1.0]:
            R_initial = self.transformer.get_rotation_between_vectors(sign * frame.pc1_axis, target_axis)
            n_steps = int(2 * np.pi / self.angle_step_rad)
            
            for i in range(n_steps):
                R_roll = self.transformer.get_rotation_about_axis(target_axis, i * self.angle_step_rad)
                R_total = R_initial @ R_roll
                
                transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
                P = self.matcher.match(transformed, ref_frame.means, tau=tau, epsilon=eps, return_matrix=True)
                
                # Soft Euclidean Cost: sum(P_ij * ||x_i - y_j||^2)
                dist_sq = np.sum((transformed[:, None, :] - ref_frame.means[None, :, :])**2, axis=2)
                cost = np.sum(P * dist_sq)

                history.append({
                    'R': R_total,
                    'sign': sign,
                    'angle': np.degrees(i * self.angle_step_rad),
                    'cost': cost
                })

        # Peak Detection: Sort by cost and filter for unique angular valleys
        sorted_h = sorted(history, key=lambda x: x['cost'])
        unique_valleys = []
        for entry in sorted_h:
            is_new_valley = True
            for v in unique_valleys:
                if entry['sign'] == v['sign']:
                    # Periodicity check: Shortest distance on a 360-degree circle
                    diff = abs(entry['angle'] - v['angle'])
                    if min(diff, 360 - diff) < 60: # Threshold for distinct valleys
                        is_new_valley = False
                        break
            if is_new_valley:
                unique_valleys.append(entry)
            if len(unique_valleys) >= k: break
                        
        return unique_valleys, (history if return_trace else None)
    
    def _refine_soft_icp(self, frame, ref_frame, initial_R, return_trace=False):
        """Weighted Kabsch ICP refinement using Sinkhorn correspondences."""
        R_curr, t_curr = initial_R, ref_frame.center_of_mass
        eps = self.settings.get('epsilon_refine', 0.01) # Sharper epsilon for refinement
        tau = self.settings.get('tau', 1e6)
        iters = self.settings.get('icp_iters', 10)
        
        trace_data = [] if return_trace else None
        
        for i in range(iters):
            current_pts = frame.normalized_coords @ R_curr + t_curr        
            P = self.matcher.match(current_pts, ref_frame.means, tau=tau, epsilon=eps, return_matrix=True)
            
            if return_trace:
                # Euclidean cost of the MAP (Maximum A Posteriori) assignment
                col_ind = np.argmax(P, axis=1)
                cost = np.sum((current_pts - ref_frame.means[col_ind])**2)
                trace_data.append({'iter': i, 'cost': cost})
            
            # Update transformation using Weighted Kabsch
            self.transformer.fit_weighted(frame.normalized_coords, ref_frame.means, P)
            R_curr, t_curr = self.transformer.R, self.transformer.t
            
        return R_curr, t_curr, trace_data
# class LegacyEngine:
#     def __init__(self, atlas, slice_db, matcher, transformer, settings: dict = None):
#         self.atlas = atlas
#         self.slice_db = slice_db
#         self.matcher = matcher
#         self.transformer = transformer
#         self.settings = settings if settings is not None else {
#             'angle_step_deg': 4.0,
#             'icp_iters': 5
#         }
#         self.angle_step_rad = np.radians(self.settings['angle_step_deg'])

#     def align_frame(self, frame, trace=False):
#         """Standardizes and aligns an experimental frame against biological hypotheses."""
#         frame.prepare()
        
#         # Candidate selection
#         candidate_ids = self.slice_db.get_candidates(len(frame))
#         best_overall_result = None
        
#         # Trace to map loss landscape
#         landscape_traces = {}
        
#         tau = self.settings.get('tau', 1.0)
#         epsilon = self.settings.get('epsilon', 0.05)
#         for s_id in candidate_ids:
#             # Build slice models
#             labels = self.slice_db.get_labels(s_id)
#             ref_frame = ReferenceFrame(labels, self.atlas)
            
#             # Coarse scan (Hungarian or sinkhorn)
#             if isinstance(self.matcher, SinkhornMatcher):
#                 best_R_init, _, coarse_history = self._run_coarse_scan_sinkhorn(frame, ref_frame, return_trace=trace)
#             else:
#                 best_R_init, _, coarse_history = self._run_coarse_scan(frame, ref_frame, return_trace=trace)
            
#             # ICP Refinement
#             refined_R, refined_t, icp_history = self._refine_icp(frame, ref_frame, best_R_init, return_trace=trace)
            
#             # Label and score
#             aligned_coords = frame.normalized_coords @ refined_R + refined_t
#             final_cost, assignments = self._final_mah_score(aligned_coords, ref_frame, tau=tau)
#             # Map indices to labels, handling the 'Slack' rejection index
#             # If an assignment index is >= n_real, it was matched to the slack bin
#             final_labels = []
#             for idx in assignments:
#                 if idx < ref_frame.n_real:
#                     final_labels.append(ref_frame.labels[idx])
#                 else:
#                     final_labels.append("unassigned")
#             # Log trace if requested
#             if trace:
#                 landscape_traces[s_id] = {
#                     'coarse': coarse_history,
#                     'icp_history': icp_history,
#                     'final_mahalanobis': final_cost
#                 }
            
#             # Track winner
#             if best_overall_result is None or final_cost < best_overall_result['cost']:
#                 best_overall_result = {
#                     'slice_id': s_id,
#                     'cost': final_cost,
#                     'labels': final_labels,
#                     'coords': aligned_coords,
#                     'scale_factor': frame.scale_factor,
#                     'settings_snapshot': self.settings.copy()
#                 }
            
#         if trace:
#                 return best_overall_result, landscape_traces
            
#         return best_overall_result
        
#     def _run_coarse_scan(self, frame, ref_frame, return_trace=False):
#         """PC1 based rotation scan."""
#         best_cost = float('inf')
#         best_R = None
#         eps = self.settings.get('epsilon', 0.05) #
#         tau = self.settings.get('tau', 1.0)
#         # Trace storage
#         trace_data = [] if return_trace else None
        
#         target_axis = ref_frame.pc1_axis
#         # Scan both PC1 orientations
#         for sign in [+1.0, -1.0]:
#             R_initial = self.transformer.get_rotation_between_vectors(
#                 sign * frame.pc1_axis, target_axis
#             )
            
#             n_steps = int(2 * np.pi / self.angle_step_rad)
#             for i in range(n_steps):
#                 R_roll = self.transformer.get_rotation_about_axis(
#                     target_axis, i * self.angle_step_rad
#                 )
#                 R_total = R_initial @ R_roll
                
#                 # Center 
#                 transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
#                 _, col_ind = self.matcher.match(transformed, ref_frame.means, tau=tau, epsilon=eps)
                
#                 # Cost must be calculated in absolute atlas space
#                 diff = transformed - ref_frame.means[col_ind]
#                 cost = np.sum(diff**2) 

#                 # Record trace if requested
#                 if return_trace:
#                     trace_data.append({
#                         'sign': sign,
#                         'angle_deg': np.degrees(i * self.angle_step_rad),
#                         'cost': cost
#                     })
                
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_R = R_total
                        
#         return best_R, best_cost, trace_data
    
#     def _run_coarse_scan_sinkhorn(self, frame, ref_frame, return_trace=False):
#         """PC1 based rotation scan using Sinkhorn for landscape smoothing."""
#         best_cost = float('inf')
#         best_R = None
        
#         # Pull settings
#         eps = self.settings.get('epsilon_coarse', 0.01) # Usually higher epsilon for coarse
#         tau = self.settings.get('tau', 1e6)
        
#         trace_data = [] if return_trace else None
#         target_axis = ref_frame.pc1_axis

#         for sign in [+1.0, -1.0]:
#             R_init = self.transformer.get_rotation_between_vectors(sign * frame.pc1_axis, target_axis)
#             n_steps = int(2 * np.pi / self.angle_step_rad)
            
#             for i in range(n_steps):
#                 R_roll = self.transformer.get_rotation_about_axis(target_axis, i * self.angle_step_rad)
#                 R_total = R_init @ R_roll
#                 transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
                
#                 # Get asisgnment matrix
#                 P = self.matcher.match(transformed, ref_frame.means, tau=tau, epsilon=eps, return_matrix=True)
#                 # Compute sinkhorn weighted cost
#                 # sum(P_ij * ||x_i - y_j||^2)
#                 dist_sq = np.sum((transformed[:, None, :] - ref_frame.means[None, :, :])**2, axis=2)
#                 cost = np.sum(P * dist_sq)
                
#                 if return_trace:
#                     trace_data.append({'sign': sign, 'angle_deg': np.degrees(i * self.angle_step_rad), 'cost': cost})
                
#                 if cost < best_cost:
#                     best_cost, best_R = cost, R_total
        
#         return best_R, best_cost, trace_data
    
#     def _refine_icp(self, frame, ref_frame, initial_R, return_trace=False):
#         """Snap alignment into place with Euclidean ICP."""
#         R_curr = initial_R
#         t_curr = ref_frame.center_of_mass
#         # Pull dynamic settings for this iteration
#         tau = self.settings.get('tau', 1.0)
#         eps = self.settings.get('epsilon', 0.05)
#         icp_iters = self.settings.get('icp_iters', 5)
#         # Trace
#         trace_data =[] if return_trace else None
        
#         for i in range(icp_iters):
#             #epsilon = self.settings.get('epsilon', 0.05)
#             current_pts = frame.normalized_coords @ R_curr + t_curr        
#             # Hybrid Assignment
#             if hasattr(self.matcher, 'compute_P'):
#                 # Sinkhorn Path
#                 W = self.matcher.match(current_pts, ref_frame.means, tau=tau, epsilon=eps, return_matrix = True)
#                 # For the TRACE , we take the Maximum A Posteriori (MAP) neighbor
#                 col_ind = np.argmax(W, axis=1)
#             else:
#                 # Hungarian path
#                 _, col_ind = self.matcher.match(current_pts, ref_frame.means, tau=tau)
#                 W = np.zeros((len(current_pts), len(ref_frame.means)))
#                 W[np.arange(len(current_pts)), col_ind] = 1.0
#             # Trace optionally
#             if return_trace:
#                 # Calculate Euclidean cost for this iteration
#                 diff = current_pts - ref_frame.means[col_ind]
#                 cost = np.sum(diff**2)
#                 trace_data.append({'iteration': i, 'cost': cost})
#             # Calculate rigid transform for current correspondences
#             self.transformer.fit_weighted(frame.normalized_coords, ref_frame.means, W)
#             R_curr, t_curr = self.transformer.R, self.transformer.t
            
#         return R_curr, t_curr, trace_data
    
#     def _final_mah_score(self, aligned_coords, ref_frame, tau=1.0):
#         """Vectorized Mahalanobis scoring for final label assignment."""
#         N = len(aligned_coords)
#         M = ref_frame.n_real
        
#         D = np.zeros((N, M))
#         for i in range(N):
#             mu = ref_frame.means[i]
#             inv_cov = ref_frame.inv_covs[i]
#             diff = aligned_coords - mu
#             # Mahalanobis: (x-mu)^T * InvCov * (x-mu)
#             D[:, i] = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        
#         total_size = N + M
#         D_aug = np.full((total_size, total_size), tau)
#         D_aug[:N, :M] = D
#         D_aug[N:, M:] = 0
        
#         row_ind_full, col_ind_full = linear_sum_assignment(D_aug)
        
#         # We only care about the assignments for our N real observations
#         final_assignments = col_ind_full[:N]
#         total_cost = D_aug[np.arange(N), final_assignments].sum()
        
#         return total_cost, final_assignments