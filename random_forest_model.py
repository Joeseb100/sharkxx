"""
Random Forest Classifier for Shark Habitat Modeling
Includes training, validation, and prediction capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

class SharkHabitatModel:
    """Random Forest model for shark habitat suitability"""
    
    def __init__(self, n_estimators=500, max_depth=20, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Handle any class imbalance
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        self.feature_names = None
        self.trained = False
        
    def train(self, X, y, feature_names=None):
        """Train the Random Forest model"""
        print("=" * 80)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("=" * 80)
        
        self.feature_names = feature_names if feature_names else X.columns.tolist()
        
        print(f"\nTraining data:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Presence: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"  Absence: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
        
        print(f"\nModel parameters:")
        print(f"  n_estimators: {self.model.n_estimators}")
        print(f"  max_depth: {self.model.max_depth}")
        print(f"  class_weight: {self.model.class_weight}")
        
        print("\nTraining Random Forest...")
        self.model.fit(X, y)
        self.trained = True
        
        # Training accuracy
        train_score = self.model.score(X, y)
        print(f"\n✓ Training completed!")
        print(f"  Training accuracy: {train_score:.3f}")
        
        return self.model
    
    def cross_validate(self, X, y, cv=5):
        """Perform k-fold cross-validation"""
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION")
        print("=" * 80)
        
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        # Stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Multiple metrics
        metrics = {
            'accuracy': cross_val_score(self.model, X, y, cv=skf, scoring='accuracy'),
            'roc_auc': cross_val_score(self.model, X, y, cv=skf, scoring='roc_auc'),
            'precision': cross_val_score(self.model, X, y, cv=skf, scoring='precision'),
            'recall': cross_val_score(self.model, X, y, cv=skf, scoring='recall'),
            'f1': cross_val_score(self.model, X, y, cv=skf, scoring='f1')
        }
        
        print("\n✓ Cross-validation results:")
        for metric_name, scores in metrics.items():
            print(f"  {metric_name:12s}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        return metrics
    
    def get_feature_importance(self, plot=True, save_path=None):
        """Analyze feature importance"""
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature importance rankings:")
        for idx, row in importance.iterrows():
            print(f"  {row['feature']:25s}: {row['importance']:.4f}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
            plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n✓ Feature importance plot saved: {save_path}")
            else:
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                print(f"\n✓ Feature importance plot saved: feature_importance.png")
            plt.close()
        
        return importance
    
    def evaluate(self, X, y, y_pred=None, save_path=None):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        if y_pred is None:
            y_pred = self.model.predict(X)
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Absence', 'Presence']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.3f}")
        
        # Plot ROC curve
        if save_path or True:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            axes[0].set_xlabel('False Positive Rate', fontsize=12)
            axes[0].set_ylabel('True Positive Rate', fontsize=12)
            axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            axes[1].plot(recall, precision, linewidth=2, color='orange')
            axes[1].set_xlabel('Recall', fontsize=12)
            axes[1].set_ylabel('Precision', fontsize=12)
            axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            save_file = save_path if save_path else 'model_evaluation.png'
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"\n✓ Evaluation plots saved: {save_file}")
            plt.close()
        
        return {
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def predict_probability(self, X):
        """Predict probability of shark presence"""
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        # Return probability of presence (class 1)
        proba = self.model.predict_proba(X)[:, 1]
        return proba
    
    def predict(self, X, threshold=0.5):
        """Predict shark presence/absence"""
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        proba = self.predict_probability(X)
        predictions = (proba >= threshold).astype(int)
        return predictions
    
    def save_model(self, filepath='shark_habitat_model.pkl'):
        """Save trained model to disk"""
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"\n✓ Model saved: {filepath}")
    
    def load_model(self, filepath='shark_habitat_model.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.trained = True
        print(f"\n✓ Model loaded: {filepath}")
        return self.model

if __name__ == "__main__":
    print("Random Forest Classifier Module")
    print("Use in conjunction with data_preparation.py")
