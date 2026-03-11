import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class DiagnosticLayer:
    """
    DiagnosticLayer: A decoupled post-processor for alignment engines.
    Takes the diagnostics DataFrame output by the ModularAlignmentEngine 
    and predicts a confidence score for each cell assignment.
    """
    def __init__(self, model_path: str = None, training_data: pd.DataFrame = None):
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
            self.model = None
            print("DiagnosticLayer: Initialized without model. Predictions will be bypassed until trained.")

    def _train_model(self, df: pd.DataFrame) -> Pipeline:
        """Builds a robust RF pipeline with imputation and feature scaling."""
        # Ensure all expected columns exist, fill with NaN if missing entirely
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = np.nan
                
        X = df[self.numeric_features]
        y = df['is_correct'].astype(int)
        
        # The SimpleImputer prevents crashes if features (like entropy) are missing 
        # because the user ran a hard Hungarian configuration.
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        pipeline.fit(X, y)
        return pipeline

    def predict_confidence(self, result_dict: dict) -> dict:
        """
        Ingests the engine's output dictionary, calculates confidence scores 
        using the embedded 'diagnostics' DataFrame, and aggregates a frame-level score.
        """
        # Bypass if no model is loaded or no diagnostics were generated
        if not self.model or 'diagnostics' not in result_dict:
            return result_dict
            
        diag_df = result_dict['diagnostics']
        
        # Safely pad missing features with NaN for the imputer to catch
        for col in self.numeric_features:
            if col not in diag_df.columns:
                diag_df[col] = np.nan
                
        X = diag_df[self.numeric_features]
        
        # Predict probability of class 1 (is_correct == True)
        probs = self.model.predict_proba(X)[:, 1]
        
        # Inject cell-level predictions back into the DataFrame
        diag_df['confidence_score'] = probs
        
        # Aggregate and inject frame-level confidence into the main result dictionary
        result_dict['mean_confidence'] = float(np.mean(probs))
        result_dict['diagnostics'] = diag_df
        
        return result_dict

    def save_model(self, path: str):
        """Exports the trained model to disk."""
        if self.model:
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Returns a DF of feature weights for publication figures/tables."""
        if not self.model: return pd.DataFrame()
        importances = self.model.named_steps['rf'].feature_importances_
        return pd.DataFrame({
            'feature': self.numeric_features,
            'importance': importances
        }).sort_values('importance', ascending=False)

    def get_performance_summary(self, val_df: pd.DataFrame) -> dict:
        """Generates standard classification metrics."""
        if not self.model: return {}
        
        for col in self.numeric_features:
            if col not in val_df.columns:
                val_df[col] = np.nan
                
        X = val_df[self.numeric_features]
        y_true = val_df['is_correct'].astype(int)
        y_pred = self.model.predict(X)
        return classification_report(y_true, y_pred, output_dict=True)