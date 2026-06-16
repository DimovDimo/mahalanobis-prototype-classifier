import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# ML Preprocessing & Frameworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Model Persistence
import joblib

class StellarPipeline:
    def __init__(self, dataset_uid="fedesoriano/stellar-classification-dataset-sdss17"):
        """Initializes the stellar classification automation pipeline."""
        self.dataset_uid = dataset_uid
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.metrics_df = None
        self.ensemble = None
        self.essential_features = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
        
        # Configure visualization parameters
        warnings.filterwarnings('ignore')
        sns.set_theme(style="darkgrid")
        
    def load_and_clean_data(self):
        """Downloads from Kaggle API, applies structural noise filtration, and keeps core features."""
        print("[!] Activating Data Acquisition via KaggleHub...")
        download_path = kagglehub.dataset_download(self.dataset_uid)
        
        csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("Could not locate source CSV in download directory.")
            
        target_csv = os.path.join(download_path, csv_files[0])
        self.df = pd.read_csv(target_csv)
        print(f"[+] Initial Dataset Loaded. Dimensions: {self.df.shape}")
        
        # Clean spectral extreme outlier noise points
        filters = ['u', 'g', 'r', 'i', 'z']
        self.df = self.df[
            (self.df['u'] > 0) & (self.df['g'] > 0) & 
            (self.df['r'] > 0) & (self.df['i'] > 0) & (self.df['z'] > 0)
        ]
        
        # Isolate physical features and target class
        self.df = self.df[self.essential_features + ['class']]
        print(f"[+] Outlier removal completed. Retained Dimensions: {self.df.shape}")
        
    def prepare_datasets(self, test_size=0.20, random_state=42):
        """Splits the features using stratified sampling and applies data leakage protection scaling."""
        print("[!] Executing Dataset Split and Scaler Normalization...")
        X = self.df[self.essential_features]
        y = self.df['class']
        
        # Stratified partition distribution allocation
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standard scaling transformations avoiding data leakage
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        # Re-map back to Pandas DataFrames for consistent down-stream feature naming rules
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.essential_features)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.essential_features)
        print("[+] Datasets prepared successfully with proper scaling limits.")
        
    def run_benchmark(self):
        """Trains individual classifiers and builds a structured leaderboard dynamically."""
        print("[!] Commencing Multi-Model Benchmark Ingestion...")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
            'CatBoost': CatBoostClassifier(iterations=150, learning_rate=0.1, depth=6, random_seed=42, verbose=0),
            'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM': LinearSVC(dual=False, max_iter=2000, random_state=42)
        }
        
        performance_log = []
        
        for name, model in models.items():
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            duration = time.time() - start_time
            
            preds = model.predict(self.X_test)
            if preds.ndim > 1:
                preds = preds.ravel()
                
            acc = accuracy_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds, average='macro')
            
            performance_log.append({
                'Model Name': name,
                'Accuracy': acc,
                'F1-Score (Macro)': f1,
                'Training Time (sec)': round(duration, 3)
            })
            print(f" -> Finished {name:25} | Accuracy: {acc:.5f} | Time: {duration:.2f}s")
            
        self.metrics_df = pd.DataFrame(performance_log)
        
    def train_voting_ensemble(self):
        """Constructs and injects a soft voting ensemble into the active tracking records."""
        print("[!] Fitting Hybrid Soft Voting Ensemble Classifier...")
        rf_base = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        lgb_base = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        cat_base = CatBoostClassifier(iterations=150, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
        
        self.ensemble = VotingClassifier(
            estimators=[('rf', rf_base), ('lgb', lgb_base), ('cat', cat_base)],
            voting='soft', n_jobs=-1
        )
        
        start_time = time.time()
        self.ensemble.fit(self.X_train, self.y_train)
        duration = time.time() - start_time
        
        preds = self.ensemble.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average='macro')
        
        ensemble_row = pd.DataFrame([{
            'Model Name': 'Voting Ensemble (Hybrid)', 'Accuracy': acc,
            'F1-Score (Macro)': f1, 'Training Time (sec)': round(duration, 3)
        }])
        
        self.metrics_df = pd.concat([self.metrics_df, ensemble_row], ignore_index=True)
        print(f"[+] Ensemble Complete | Final Combined Accuracy: {acc:.5f}")
        
    def plot_final_leaderboard(self):
        """Renders the comprehensive performance chart cleanly without warning messages."""
        final_sorted = self.metrics_df.sort_values(by='Accuracy', ascending=False)
        
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(
            x='Accuracy', y='Model Name', data=final_sorted, 
            hue='Model Name', palette='Set2', legend=False
        )
        
        for p in ax.patches:
            width = p.get_width()
            if width > 0:
                ax.annotate(f'{width:.5f}', (width, p.get_y() + p.get_height() / 2.),
                            ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')
                            
        plt.title('All-Inclusive Pipeline Performance Accuracy Metric Dashboard', fontsize=13, pad=15)
        plt.xlabel('Classification Accuracy Score')
        plt.ylabel('Evaluated Architectures')
        plt.xlim(0.9, 1.0)
        plt.show()
        
    def export_production_artifacts(self, model_filename="best_stellar_ensemble.joblib", scaler_filename="stellar_scaler.joblib"):
        """Saves the top-performing ensemble and scaler objects into binary storage formats."""
        print(f"[!] Exporting Model State to '{model_filename}' and Scaler State to '{scaler_filename}'...")
        joblib.dump(self.ensemble, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        print("[+] Persistence workflow successfully locked and saved to file storage.")

# Verification execution sequence triggering the full pipeline programmatically
if __name__ == "__main__":
    pipeline = StellarPipeline()
    pipeline.load_and_clean_data()
    pipeline.prepare_datasets()
    pipeline.run_benchmark()
    pipeline.train_voting_ensemble()
    pipeline.plot_final_leaderboard()
    pipeline.export_production_artifacts()
