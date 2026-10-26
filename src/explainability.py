import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

DATA_PATH = 'data/processed/test_processed.csv'
MODEL_PATH = 'models/baseline_rf.pkl'
# --- STEP 1: LOAD DATA AND MODEL ---
# Raja, this part loads the Random Forest model that I trained.
# You don't need to change this.
def load_data_and_model():
    print("load trained model")
    test_df = pd.read_csv(DATA_PATH)
    X_test = test_df.drop('income',axis=1)
    model = joblib.load(MODEL_PATH)
    return X_test, model
if __name__=="__main__":
    X_test, model = load_data_and_model()
    print("data loaded successfully:",X_test.shape)
    print("model loaded successfully")

# --- STEP 2: SHAP (GLOBAL EXPLAINABILITY) ---
# Raja, this code calculates the SHAP values. 
# Run this, and it will save a beautiful summary plot in reports/figures/shap_summary.png.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values,X_test, show=False)
    plt.savefig('reports/figures/shap_summary.png')
    plt.close()
    print("shap plot saved to reports/figures/shap_summary.png")
# --- STEP 3: LIME (LOCAL EXPLAINABILITY) ---
# Raja, this part picks one specific person (line 5 of the test data).
# It explains why the model made its decision and saves it as an HTML file.
    print("calculating for 5 instances")
    explainer = lime.lime_tabular.LimeTabularExplainer(X_test.values,feature_names=X_test.columns,class_names=['<=50K', '>50K'],mode='classification')
    instance = X_test.iloc[5]
    exp = explainer.explain_instance(instance.values, model.predict_proba, num_features=5)
    exp.save_to_file('reports/figures/lime_explanation.html')
    print("lime explanation saved to reports/figures/lime_explanation.html")