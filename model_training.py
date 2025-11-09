import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the Parkinson's dataset"""
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    
    # Separate features and target
    X = df.drop(['name', 'status'], axis=1)
    y = df['status']
    
    return X, y

def create_advanced_ensemble_model():
    """Create an advanced ensemble model with hyperparameter tuning"""
    
    # Individual models with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=3,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    extra_trees = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='auto',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    
    # Create voting classifier with optimized weights
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb),
            ('extra', extra_trees),
            ('svm', svm)
        ],
        voting='soft',
        weights=[3, 2, 3, 2, 1]  # Optimized weights based on individual performance
    )
    
    return ensemble

def train_model():
    """Train the advanced ensemble model"""
    print("=" * 60)
    print("ADVANCED PARKINSON'S VOICE DETECTION MODEL TRAINING")
    print("=" * 60)
    
    print("\n📊 Loading data...")
    X, y = load_and_prepare_data()
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n🤖 Training advanced ensemble model...")
    print("-" * 40)
    
    model = create_advanced_ensemble_model()
    
    # Cross-validation with detailed metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train), 1):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train fold model
        fold_model = create_advanced_ensemble_model()
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Evaluate
        y_pred = fold_model.predict(X_fold_val)
        y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        
        cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        cv_scores['precision'].append(precision_score(y_fold_val, y_pred))
        cv_scores['recall'].append(recall_score(y_fold_val, y_pred))
        cv_scores['f1'].append(f1_score(y_fold_val, y_pred))
        cv_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
        
        print(f"Fold {fold}: Acc={cv_scores['accuracy'][-1]:.4f}, "
              f"F1={cv_scores['f1'][-1]:.4f}, "
              f"AUC={cv_scores['roc_auc'][-1]:.4f}")
    
    print("-" * 40)
    print("\n📈 Cross-Validation Results:")
    for metric, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric.upper():<12}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    # Train final model on full training set
    print("\n🎯 Training final model on full training set...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    
    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    for metric, value in test_metrics.items():
        print(f"{metric:<12}: {value:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Feature importance
    if hasattr(model, 'estimators_'):
        print("\n📊 Top 10 Important Features:")
        # Get feature importance from Random Forest component
        rf_model = model.estimators_[0][1]
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']:<25}: {row['importance']:.4f}")
    
    # Save model and preprocessors
    print("\n💾 Saving model and preprocessors...")
    import os
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/ensemble_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    
    print("✅ Model saved successfully!")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, scaler

if __name__ == "__main__":
    train_model()