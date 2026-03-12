import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from aligner.models import ReferenceFrame
from aligner.atlas import SliceTimeAtlas, GPToStaticAdapter
from aligner.config import InitStrategy, AtlasStrategy, MatcherType

class ModularAlignmentEngine:
    """
    A unified alignment engine that uses composition and a PipelineConfig 
    to dictate its strategy.
    """
    def __init__(self, config, atlas, slice_db, coarse_matcher, icp_matcher, transformer):
        self.config = config
        self.atlas = atlas
        self.slice_db = slice_db
        self.coarse_matcher = coarse_matcher
        self.icp_matcher = icp_matcher
        self.transformer = transformer
        self.angle_step_rad = np.radians(self.config.angle_step_deg)
        
        # Automatically wrap GP atlases for time-resolved lookups
        if self.config.atlas_strategy == AtlasStrategy.TIME_RESOLVED:
            self.hybrid_atlas = SliceTimeAtlas(
                self.atlas, 
                self.slice_db
                )
        else:
            self.hybrid_atlas = None

    def _get_verified_candidates(self, n_cells):
        """Checks if the SliceAtlas has biological hypotheses for this cell count."""
        candidates = self.slice_db.get_candidates(n_cells)
        if not candidates:
            print(f"[Atlas Gap] No biological slices found for N={n_cells} cells.")
            return None
        return candidates

    def _final_mah_score(self, aligned_coords, ref_frame):
        N, M = len(aligned_coords), ref_frame.n_real
        D = np.zeros((N, M))
        for j in range(M):
            diff = aligned_coords - ref_frame.means[j]
            D[:, j] = np.einsum('ij,jk,ik->i', diff, ref_frame.inv_covs[j], diff)
        
        if self.config.use_slack:
            C_aug = np.full((N + M, N + M), self.config.tau)
            C_aug[:N, :M] = D
            C_aug[N:, M:] = 0
            row, col = linear_sum_assignment(C_aug)
            return C_aug[np.arange(N), col[:N]].sum(), col[:N], D
        else:
            # Strict assignment based purely on Mahalanobis distance
            row, col = linear_sum_assignment(D)
            return D[row, col].sum(), col, D

    # def _coarse_scan(self, frame, ref_frame, k=1, return_trace=False):
    #     history = []
    #     target_axis = ref_frame.pc1_axis
        
    #     # Setup kwargs dynamically based on the chosen coarse matcher
    #     kwargs = {'tau': self.config.tau, 'return_matrix': True}
    #     if self.config.coarse_matcher == MatcherType.SINKHORN:
    #         kwargs['epsilon'] = self.config.epsilon_coarse
            
    #     for sign in [+1.0, -1.0]:
    #         R_initial = self.transformer.get_rotation_between_vectors(sign * frame.pc1_axis, target_axis)
    #         n_steps = int(2 * np.pi / self.angle_step_rad)
            
    #         for i in range(n_steps):
    #             R_roll = self.transformer.get_rotation_about_axis(target_axis, i * self.angle_step_rad)
    #             R_total = R_initial @ R_roll
                
    #             transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
    #             dist_sq_mat = np.sum((transformed[:, None, :] - ref_frame.means[None, :, :])**2, axis=2)
                
    #             # Get the P matrix (Binary for Hungarian, Probabilities for Sinkhorn)
    #             P = self.coarse_matcher.match(transformed, ref_frame.means, **kwargs)
                
    #             # Element-wise multiplication of assignments and distances
    #             cost = np.sum(P[:len(transformed), :ref_frame.n_real] * dist_sq_mat)

    #             history.append({
    #                 'R': R_total, 'sign': sign,
    #                 'angle': np.degrees(i * self.angle_step_rad), 'cost': cost
    #             })

    #     sorted_h = sorted(history, key=lambda x: x['cost'])
    #     unique_valleys = []
    #     for entry in sorted_h:
    #         is_new_valley = True
    #         for v in unique_valleys:
    #             if entry['sign'] == v['sign']:
    #                 diff = abs(entry['angle'] - v['angle'])
    #                 if min(diff, 360 - diff) < 60:
    #                     is_new_valley = False
    #                     break
    #         if is_new_valley:
    #             unique_valleys.append(entry)
    #         if len(unique_valleys) >= k: break
                        
    #     return unique_valleys, (history if return_trace else None)

    # def _refine_icp(self, frame, ref_frame, initial_R, return_trace=False):
    #     """Weighted Kabsch ICP. Works with both Hard (Hungarian) and Soft (Sinkhorn) Matchers."""
    #     R_curr, t_curr = initial_R, ref_frame.center_of_mass
    #     trace_data = [] if return_trace else None
        
    #     # Inject Sinkhorn epsilon only if Sinkhorn is the active matcher
    #     kwargs = {'tau': self.config.tau, 'return_matrix': True}
    #     if self.config.icp_matcher == MatcherType.SINKHORN:
    #         kwargs['epsilon'] = self.config.epsilon_refine
            
    #     for i in range(self.config.icp_iters):
    #         current_pts = frame.normalized_coords @ R_curr + t_curr        
    #         P = self.icp_matcher.match(current_pts, ref_frame.means, **kwargs)
            
    #         if return_trace:
    #             col_ind = np.argmax(P, axis=1)
    #             cost = np.sum((current_pts - ref_frame.means[col_ind])**2)
    #             trace_data.append({'iter': i, 'cost': cost})
            
    #         # Transformer accepts both binary matrices (Hard) and probability matrices (Soft)
    #         self.transformer.fit_weighted(frame.normalized_coords, ref_frame.means, P)
    #         R_curr, t_curr = self.transformer.R, self.transformer.t
            
    #     return R_curr, t_curr, trace_data
    
    def _coarse_scan(self, frame, ref_frame, k=1, return_trace=False):
        history = []
        target_axis = ref_frame.pc1_axis
        
        # FIX 2: Explicitly pass 'use_slack' to the matcher
        kwargs = {'tau': self.config.tau, 'use_slack': self.config.use_slack, 'return_matrix': True}
        if self.config.coarse_matcher == MatcherType.SINKHORN:
            kwargs['epsilon'] = self.config.epsilon_coarse
            
        for sign in [+1.0, -1.0]:
            R_initial = self.transformer.get_rotation_between_vectors(sign * frame.pc1_axis, target_axis)
            n_steps = int(2 * np.pi / self.angle_step_rad)
            
            for i in range(n_steps):
                R_roll = self.transformer.get_rotation_about_axis(target_axis, i * self.angle_step_rad)
                R_total = R_initial @ R_roll
                
                transformed = frame.normalized_coords @ R_total + ref_frame.center_of_mass
                dist_sq_mat = np.sum((transformed[:, None, :] - ref_frame.means[None, :, :])**2, axis=2)
                
                P = self.coarse_matcher.match(transformed, ref_frame.means, **kwargs)
                
                # FIX 3: Apply the tau penalty to any probability mass dumped to slack
                assigned_cost = np.sum(P[:len(transformed), :ref_frame.n_real] * dist_sq_mat)
                unassigned_count = len(transformed) - np.sum(P[:len(transformed), :ref_frame.n_real])
                cost = assigned_cost + (unassigned_count * self.config.tau)

                history.append({
                    'R': R_total, 'sign': sign,
                    'angle': np.degrees(i * self.angle_step_rad), 'cost': cost
                })

        sorted_h = sorted(history, key=lambda x: x['cost'])
        unique_valleys = []
        for entry in sorted_h:
            is_new_valley = True
            for v in unique_valleys:
                if entry['sign'] == v['sign']:
                    diff = abs(entry['angle'] - v['angle'])
                    if min(diff, 360 - diff) < 60:
                        is_new_valley = False
                        break
            if is_new_valley:
                unique_valleys.append(entry)
            if len(unique_valleys) >= k: break
                        
        return unique_valleys, (history if return_trace else None)

    def _refine_icp(self, frame, ref_frame, initial_R, return_trace=False):
        """Weighted Kabsch ICP. Works with both Hard (Hungarian) and Soft (Sinkhorn) Matchers."""
        R_curr, t_curr = initial_R, ref_frame.center_of_mass
        trace_data = [] if return_trace else None
        
        # FIX 4: Explicitly pass 'use_slack' to the matcher
        kwargs = {'tau': self.config.tau, 'use_slack': self.config.use_slack, 'return_matrix': True}
        if self.config.icp_matcher == MatcherType.SINKHORN:
            kwargs['epsilon'] = self.config.epsilon_refine
            
        for i in range(self.config.icp_iters):
            current_pts = frame.normalized_coords @ R_curr + t_curr        
            P = self.icp_matcher.match(current_pts, ref_frame.means, **kwargs)
            
            if return_trace:
                col_ind = np.argmax(P, axis=1)
                cost = np.sum((current_pts - ref_frame.means[col_ind])**2)
                trace_data.append({'iter': i, 'cost': cost})
            
            self.transformer.fit_weighted(frame.normalized_coords, ref_frame.means, P)
            R_curr, t_curr = self.transformer.R, self.transformer.t
            
        return R_curr, t_curr, trace_data

    def align_frame(self, frame, life_history_df=None, trace=False, return_diagnostics=False):
        """Standardizes and aligns an experimental frame against biological hypotheses."""
        frame.prepare()
        candidate_ids = self._get_verified_candidates(len(frame))
        if candidate_ids is None: return None
        
        best_overall_result = None
        landscape_traces = {} 
        k_tourn = self.config.k_tournament if self.config.init_strategy == InitStrategy.TOURNAMENT else 1
        
        for s_id in candidate_ids:
            # 1. Hypothesis Generation (Static vs Time-Resolved)
            if self.config.atlas_strategy == AtlasStrategy.TIME_RESOLVED:
                state = self.hybrid_atlas.get_temporal_state(s_id, time_offset=0.5)
                if not state: continue
                adapter = GPToStaticAdapter(state['labels'], state['means'], state['variances'])
                ref_frame = ReferenceFrame(state['labels'], adapter)
            else:
                labels = self.slice_db.get_labels(s_id)
                ref_frame = ReferenceFrame(labels, self.atlas)
            
            # 2. Coarse Scan
            valleys, coarse_history = self._coarse_scan(frame, ref_frame, k=k_tourn, return_trace=trace)
            
            tournament_outcomes = []
            for i, init in enumerate(valleys):
                # 3. Refinement ICP (Hard or Soft based on Matcher)
                refined_R, refined_t, icp_history = self._refine_icp(frame, ref_frame, init['R'], return_trace=trace)
                aligned_coords = frame.normalized_coords @ refined_R + refined_t
                
                # 4. Final Scoring
                cost, assignments, D = self._final_mah_score(aligned_coords, ref_frame)
                
                final_labels = [ref_frame.labels[idx] if idx < ref_frame.n_real else "unassigned" for idx in assignments]
                
                outcome = {
                    'slice_id': s_id, 'cost': cost, 'coords': aligned_coords,
                    'labels': final_labels, 'init_angle': init['angle'], 'start_rank': i + 1
                }
                
                # Pre-compute diagnostic features if enabled
                if self.config.enable_diagnostics:
                    kwargs = {'tau': self.config.tau, 'use_slack': self.config.use_slack, 'return_matrix': True}
                    if self.config.icp_matcher == MatcherType.SINKHORN: kwargs['epsilon'] = self.config.epsilon_refine
                    P = self.icp_matcher.match(aligned_coords, ref_frame.means, **kwargs)
                    outcome['per_cell_costs'] = np.sum(P[:len(aligned_coords), :ref_frame.n_real] * D, axis=1)
                    outcome['entropy'] = entropy(P[:len(aligned_coords), :ref_frame.n_real] + 1e-12, axis=1)

                tournament_outcomes.append(outcome)
                
                if best_overall_result is None or cost < best_overall_result['cost']:
                    best_overall_result = outcome.copy()
                    best_overall_result['ref_frame'] = ref_frame # Temp storage for diagnostics

            if trace:
                landscape_traces[s_id] = {'coarse': coarse_history, 'tournament': tournament_outcomes}
                
        # 5. Build Diagnostics DataFrame for the Global Winner
        if best_overall_result and self.config.enable_diagnostics and return_diagnostics:
            ref_frame = best_overall_result.pop('ref_frame')
            meta = self.slice_db.metadata.get(best_overall_result['slice_id'], {})
            map_t = meta.get('MAP_time', np.nan)
            
            # Use the ACTUAL predicted labels that were mapped to the coordinates
            predicted_labels = best_overall_result['labels']
            
            diag_df = pd.DataFrame({
                'cell_name': predicted_labels, 
                'mah_dist': best_overall_result.get('per_cell_costs', np.nan),
                'entropy': best_overall_result.get('entropy', np.nan),
                'map_time': map_t,
                'frame_is_generated': meta.get('is_augmented', False),
                'num_cells_in_frame': len(frame),
                'time_idx': getattr(frame, 'time_idx', np.nan),
                'embryo_id': getattr(frame, 'embryo_id', 'unknown')
            })
            
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
            
            # --- THE CRITICAL FIX: Row-by-Row Spatial Comparison ---
            if frame.valid_df is not None and 'cell_name' in frame.valid_df.columns:
                true_labels = frame.valid_df['cell_name'].astype(str).values
                # Compare element-wise: Does the predicted label for coord i match the GT label for coord i?
                diag_df['is_correct'] = (np.array(predicted_labels) == true_labels)
            else:
                diag_df['is_correct'] = False
            
            best_overall_result['diagnostics'] = diag_df

        return (best_overall_result, landscape_traces) if trace else best_overall_result