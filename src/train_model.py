import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def main():
    print("Loading processed data...")
    train_df = pd.read_csv('data/processed/train_processed.csv')
    test_df = pd.read_csv('data/processed/test_processed.csv')
    
    X_train = train_df.drop('income', axis=1)
    y_train = train_df['income']
    X_test = test_df.drop('income', axis=1)
    y_test = test_df['income']
    
    # Sensitive attribute: 'sex' (0 = Female, 1 = Male based on label encoding)
    # We need the sensitive feature for the Fairlearn audit
    sensitive_train = X_train['sex']
    sensitive_test = X_test['sex']
    
    print("Training Random Forest model...")
    # Using a random state for reproducibility
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model for the explainability part (SHAP/LIME)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/baseline_rf.pkl')
    print("Model saved to models/baseline_rf.pkl")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # 1. Standard Performance Metrics
    print("\n--- Standard Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    
    # 2. Fairness Metrics with Fairlearn
    print("\n--- Fairness Audit (Sensitive Attribute: Sex) ---")
    print("0 = Female, 1 = Male")
    
    # Create a MetricFrame
    metrics = {
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate,
        'false_positive_rate': false_positive_rate
    }
    
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )
    
    print("\nMetrics by Group:")
    print(mf.by_group)
    
    # Calculate Disparities
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)
    
    print("\n--- Fairness Disparities (Differences) ---")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference: {eo_diff:.4f}")
    print("\nNote: Values closer to 0 are better. Values > 0.1 indicate potential bias.")

if __name__ == "__main__":
    main()