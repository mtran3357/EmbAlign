import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from aligner.models import ReferenceFrame
from aligner.matcher import HungarianMatcher, SinkhornMatcher
from aligner.atlas import SliceTimeAtlas, GPToStaticAdapter
from scipy.stats import entropy

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
    
    def _get_verified_candidates(self, n_cells):
        """
        Check if the SliceAtlas has biological hypotheses for this cell count.
        Matches the 'N not in slices_by_N' safety check from the legacy script.
        """
        candidates = self.slice_db.get_candidates(n_cells)
        if not candidates:
            print(f"[Atlas Gap] No biological slices found for N={n_cells} cells.")
            return None
        return candidates

class LegacyEngine(BaseEngine):
    """
    LegacyEngine: Implements the original rigid-body alignment algorithm.
    1. Discrete Hungarian matching (1-to-1).
    2. Single-start refinement (Winner-take-all from coarse scan).
    3. Euclidean-only cost minimization.
    """
    
    def align_frame(self, frame, trace=False, **kwargs):
        """Standardizes and aligns an experimental frame against biological hypotheses."""
        frame.prepare()
        candidate_ids = self._get_verified_candidates(len(frame))
        if candidate_ids is None:
            return None
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

    def align_frame(self, frame, trace=False, **kwargs):
        """Coordinates the multi-start tournament across biological hypotheses."""
        frame.prepare()
        n_cells = len(frame)
        if n_cells < 10:
            k = self.settings.get('k_tournament', 3)
        else:
            k = 1
        
        candidate_ids = self._get_verified_candidates(len(frame))
        if candidate_ids is None:
            return None
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
        tau = self.settings.get('tau', 1e6) # High tau for coarse to prevent mass loss
        n_real_atlas = ref_frame.n_real
        for sign in [+1.0, -1.0]:
            R_initial = self.transformer.get_rotation_between_vectors(sign * frame.pc1_axis, target_axis)
            n_steps = int(2 * np.pi / self.angle_step_rad)
            
            for i in range(n_steps):
                R_roll = self.transformer.get_rotation_about_axis(target_axis, i * self.angle_step_rad)
                R_total = R_initial @ R_roll
                
                transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
                # hungarian matching
                dist_sq_mat = np.sum((transformed[:, None, :] - ref_frame.means[None, :, :])**2, axis=2)
                
                # Soft Euclidean Cost: sum(P_ij * ||x_i - y_j||^2)
                dist_sq = np.sum((transformed[:, None, :] - ref_frame.means[None, :, :])**2, axis=2)
                row_ind, col_ind = linear_sum_assignment(dist_sq_mat)
                cost = dist_sq_mat[row_ind, col_ind].sum()

                history.append({
                    'R': R_total, 'sign': sign,
                    'angle': np.degrees(i * self.angle_step_rad), 'cost': cost
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
    
class EngineV2(EngineV1):
    """Coordinates sinkhorn ICP and multistart tournaments from engine v1 with time-aware reference building."""
    def __init__(self, atlas, slice_db, matcher, transformer, settings=None):
        super().__init__(atlas, slice_db, matcher, transformer, settings)
        # wrap hp atlas for temporally resolved lookups
        self.hybrid_atlas = SliceTimeAtlas(self.atlas, self.slice_db)
        
    def align_frame(self, frame, trace=False):
        frame.prepare()
        candidate_sids = self._get_verified_candidates(len(frame))
        if candidate_sids is None:
            return None
        best_overall_result = None
        landscape_traces = {} # Track traces for all slices
        k_tourn = self.settings.get('k_tournament', 3)
        tau = self.settings.get('tau', 1e6)
        
        for s_id in candidate_sids:
            state = self.hybrid_atlas.get_temporal_state(s_id, time_offset=0.5)
            if not state: continue
            
            adapter = GPToStaticAdapter(state['labels'], state['means'], state['variances'])
            ref_frame = ReferenceFrame(state['labels'], adapter)
            
            # Fix 1: Ensure _find_top_k_valleys unpacking matches its 2-value return
            valleys, coarse_history = self._find_top_k_valleys(frame, ref_frame, k=k_tourn, return_trace=trace)
            
            tournament_outcomes = []
            for init in valleys:
                # Fix 2: Ensure _refine_soft_icp unpacking matches its 3-value return
                refined_R, refined_t, icp_history = self._refine_soft_icp(frame, ref_frame, init['R'], return_trace=trace)
                
                aligned_coords = frame.normalized_coords @ refined_R + refined_t
                cost, assignments = self._final_mah_score(aligned_coords, ref_frame, tau=tau)
                
                outcome = {
                    'slice_id': s_id, 'cost': cost, 'coords': aligned_coords,
                    'labels': [ref_frame.labels[idx] for idx in assignments]
                }
                tournament_outcomes.append(outcome)
                
                if best_overall_result is None or cost < best_overall_result['cost']:
                    best_overall_result = outcome

            if trace:
                landscape_traces[s_id] = {'coarse': coarse_history, 'tournament': tournament_outcomes}
        
        # Fix 3: BatchRunner expects a 2-tuple if trace=True
        if trace:
            return best_overall_result, landscape_traces
        return best_overall_result


class EngineV3(EngineV2):
    def __init__(self, atlas, slice_db, matcher, transformer, settings=None):
        # 1. Pass diagnostic_layer to self
        
        
        super().__init__(
            atlas=atlas, 
            slice_db=slice_db, 
            matcher=matcher, 
            transformer=transformer, 
            settings=settings
        )
        
        self.settings.setdefault('tau', 1e6)
        self.settings.setdefault('epsilon_refine', 0.01)
        print(f"Initialized EngineV3 with atlas: {self.atlas.__class__.__name__}")
    
    def align_frame(self, frame, life_history_df=None, trace=False, return_diagnostics=False):
        frame.prepare()
        candidate_ids = self._get_verified_candidates(len(frame))
        if candidate_ids is None: return None
        
        # 1. Initialize result containers
        best_overall_result = None
        landscape_traces = {} 
        global_meta = {
            'num_cells_in_frame': len(frame),
            'time_idx': getattr(frame, 'time_idx', np.nan),
            'embryo_id': getattr(frame, 'embryo_id', 'unknown')
        }
        
        tau = self.settings.get('tau', 1e6)
        eps = self.settings.get('epsilon_refine', 0.01)
        
        # 2. Iterate through hypotheses
        for s_id in candidate_ids:
            state = self.hybrid_atlas.get_temporal_state(s_id, time_offset=0.5)
            if not state: continue
            
            adapter = GPToStaticAdapter(state['labels'], state['means'], state['variances'])
            ref_frame = ReferenceFrame(state['labels'], adapter)
            
            # Capture Coarse Scan Traces
            valleys, coarse_history = self._find_top_k_valleys(
                frame, ref_frame, k=self.settings.get('k_tournament', 3), return_trace=trace
            )
            
            tournament_outcomes = []
            for init in valleys:
                # Capture ICP Refinement Traces
                refined_R, refined_t, icp_history = self._refine_soft_icp(
                    frame, ref_frame, init['R'], return_trace=trace
                )
                
                aligned_coords = frame.normalized_coords @ refined_R + refined_t
                
                # Scoring
                N, M = len(aligned_coords), ref_frame.n_real
                D = np.zeros((N, M))
                for j in range(M):
                    diff = aligned_coords - ref_frame.means[j]
                    D[:, j] = np.einsum('ij,jk,ik->i', diff, ref_frame.inv_covs[j], diff)
                
                P = self.matcher.match(aligned_coords, ref_frame.means, tau=tau, epsilon=eps, return_matrix=True)
                weighted_cost = np.sum(P[:N, :M] * D)
                
                outcome = {
                    'slice_id': s_id, 'cost': weighted_cost, 'coords': aligned_coords,
                    'labels': [ref_frame.labels[idx] for idx in np.argmax(P[:N, :M], axis=1)],
                    'icp_history': icp_history # Save for trace
                }
                tournament_outcomes.append(outcome)
                
                # Global winner tracking
                if best_overall_result is None or outcome['cost'] < best_overall_result['cost']:
                    best_overall_result = outcome.copy()
                    # Store intermediate diagnostic data for the winner
                    best_overall_result['per_cell_costs'] = np.sum(P[:N, :M] * D, axis=1)
                    best_overall_result['entropy'] = entropy(P[:N, :M] + 1e-12, axis=1)

            # Store slice landscape trace
            if trace:
                landscape_traces[s_id] = {'coarse': coarse_history, 'tournament': tournament_outcomes}

        # 3. Post-selection: Enrich the winner
        if best_overall_result and return_diagnostics:
            meta = self.slice_db.metadata.get(best_overall_result['slice_id'], {})
            map_t = meta.get('MAP_time', np.nan)
            is_gen = meta.get('is_augmented', False)
            diag_df = pd.DataFrame({
                'cell_name': [ref_frame.labels[i] for i in range(len(best_overall_result['labels']))],
                'mah_dist': best_overall_result['per_cell_costs'],
                'entropy': best_overall_result['entropy'],
                'map_time': map_t,
                'frame_is_generated': is_gen
            })
            
            # Injection of div_delta
            if life_history_df is not None:
                div_deltas = []
                for lbl in diag_df['cell_name']:
                    try:
                        lh = life_history_df.loc[[str(lbl).strip()]]
                        t_b, t_d = lh['mean_birth'].iloc[0], lh['mean_division'].iloc[0]
                        div_deltas.append((map_t - t_b) / (t_d - t_b) if (t_d - t_b) > 0 else 0.0)
                    except (KeyError, IndexError):
                        div_deltas.append(0.0)
                diag_df['div_delta'] = div_deltas
                diag_df['frame_is_generated'] = is_gen
            # Finalize diagnostics
            diag_df['is_correct'] = diag_df['cell_name'].apply(lambda x: x == dict(zip(frame.valid_df['cell_name'], frame.valid_df['cell_name'])).get(x))
            for col, val in global_meta.items(): diag_df[col] = val
            best_overall_result['diagnostics'] = diag_df
            
        return (best_overall_result, landscape_traces) if trace else best_overall_result
    