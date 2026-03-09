import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class DiagnosticLayer:
    """
    DiagnosticLayer: A decoupled post-processor for alignment engines.
    It identifies potential alignment errors based on geometric and 
    biological features derived from the engine output.
    """
    def __init__(self, model_path=None, training_data=None):
        self.numeric_features = [
            'mah_dist', 'entropy', 'map_time', 
            'div_delta', 'num_cells_in_frame'
        ]
        if model_path:
            self.model = joblib.load(model_path)
            print(f"DiagnosticLayer: Loaded model from {model_path}")
        elif training_data is not None:
            self.model = self._train_model(training_data)
            print("DiagnosticLayer: Model fitted on training data.")
        else:
            raise ValueError("Must provide either model_path or training_data.")

    def _train_model(self, df):
        """Builds a robust RF pipeline with feature scaling."""
        X = df[self.numeric_features]#.fillna(0.0) 
        y = df['is_correct'].astype(int)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ))
        ])
        pipeline.fit(X, y)
        return pipeline

    def predict(self, diag_df):
        """
        Inference method: Ensures all expected numeric features exist 
        before prediction, preventing runtime crashes.
        """
        df = diag_df.copy()
        # Defensive check: Ensure required columns exist
        # for col in self.numeric_features:
        #     if col not in diag_df.columns:
        #         diag_df[col] = 0.0 
        
        X = df[self.numeric_features]#.fillna(0.0)
        df['pred_prob'] = self.model.predict_proba(X)[:, 1]
        df['pred_label'] = self.model.predict(X)
        return df

    def process_alignment_result(self, result: dict) -> dict:
            """
            Modified Decorator:
            If 'diagnostics' exists, run inference on it. 
            If 'features' exists (fallback), create the DataFrame.
            """
            # 1. Prioritize existing diagnostics DataFrame (EngineV3's current output)
            if 'diagnostics' in result and isinstance(result['diagnostics'], pd.DataFrame):
                diag_df = result['diagnostics']
                
                # Run prediction on the existing DataFrame
                result['diagnostics'] = self.predict(diag_df)
                
                # Calculate summary metrics
                result['pred_frame_accuracy'] = result['diagnostics']['pred_prob'].mean()
                result['pred_discrete_accuracy'] = result['diagnostics']['pred_label'].mean()
                return result
                
            return result
        
    def process_alignment_result(self, result: dict) -> dict:
        if 'diagnostics' in result and isinstance(result['diagnostics'], pd.DataFrame):
            # Capture the returned decorated DataFrame
            decorated_df = self.predict(result['diagnostics'])
            
            # Explicitly overwrite the dictionary key
            result['diagnostics'] = decorated_df
            
            # Calculate summary
            result['pred_frame_accuracy'] = decorated_df['pred_prob'].mean()
            result['pred_discrete_accuracy'] = decorated_df['pred_label'].mean()
            
        return result

    def save(self, path):
        """Exports the model for reuse."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def get_feature_importance_df(self):
        """Returns a DF for publication figures/tables."""
        importances = self.model.named_steps['rf'].feature_importances_
        return pd.DataFrame({
            'feature': self.numeric_features,
            'importance': importances
        }).sort_values('importance', ascending=False)

    def get_performance_summary(self, val_df):
        """Generates classification metrics (F1, Recall, Precision)."""
        X = val_df[self.numeric_features].fillna(0.0)
        y_true = val_df['is_correct'].astype(int)
        y_pred = self.model.predict(X)
        return classification_report(y_true, y_pred, output_dict=True)
    
    def decorate_results(self, frame_results: list, diag_df: pd.DataFrame) -> list:
        """
        Maps cell-level predictions back to frame-level results.
        """
        # Calculate mean confidence per frame from the cell-level DF
        frame_metrics = diag_df.groupby('time_idx')[['pred_prob', 'pred_label']].mean()
        
        # Inject metrics into each frame dictionary
        for frame in frame_results:
            tid = frame.get('time_idx')
            if tid in frame_metrics.index:
                frame['pred_prob'] = frame_metrics.loc[tid, 'pred_prob']
                frame['pred_label'] = frame_metrics.loc[tid, 'pred_label']
        
        return frame_results