import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model(X, y):
    # Separate categorical & numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X[categorical_cols])

    # Create encoded DataFrame
    X_cat_df = pd.DataFrame(
        X_cat_encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Combine numeric + categorical
    X_final = pd.concat(
        [X_cat_df.reset_index(drop=True),
         X[numerical_cols].reset_index(drop=True)],
        axis=1
    )

    # Safety check
    if len(X_final) < 5:
        raise ValueError("Dataset too small. Upload at least 5 records.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 🔐 SAVE EVERYTHING NEEDED FOR PREDICTION
    joblib.dump(model, "model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(categorical_cols, "categorical_cols.pkl")
    joblib.dump(numerical_cols, "numerical_cols.pkl")

    return accuracy, report, model, encoder
