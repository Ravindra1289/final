import os
import pickle
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from config import MODEL_DIR # type: ignore


class ScaledMLPModel:
    def __init__(self, scaler, classifier):
        self.scaler = scaler
        self.classifier = classifier

        # Mirror key classifier attributes used by the UI.
        self.n_features_in_ = classifier.n_features_in_
        self.hidden_layer_sizes = classifier.hidden_layer_sizes
        self.n_outputs_ = classifier.n_outputs_
        self.n_layers_ = classifier.n_layers_
        self.n_iter_ = classifier.n_iter_
        self.loss_curve_ = classifier.loss_curve_
        self.classes_ = classifier.classes_
        self.solver = classifier.solver
        self.activation = classifier.activation

    def predict(self, X):
        return self.classifier.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.scaler.transform(X))

    def score(self, X, y):
        return self.classifier.score(self.scaler.transform(X), y)


def _balance_training_data(X, y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        return X, y

    max_count = int(np.max(class_counts))
    rng = np.random.default_rng(42)
    balanced_indices = []

    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        sampled_indices = rng.choice(class_indices, size=max_count, replace=True)
        balanced_indices.extend(sampled_indices.tolist())

    rng.shuffle(balanced_indices)
    return X[balanced_indices], y[balanced_indices]

def train_and_save_model(X, y, phase=2):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Cannot train the ANN because the dataset is empty.")

    # Splits exactly 15% exclusively for Validation Testing!
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )
    X_train, y_train = _balance_training_data(X_train, y_train)

    # Enhanced model with larger architecture and better regularization for superior ANN performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.0008,
        max_iter=4000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        alpha=0.00005,
        random_state=42
    )
    classifier.fit(X_train_scaled, y_train)
    mlp = ScaledMLPModel(scaler, classifier)

    model_path = os.path.join(MODEL_DIR, f"ann_model_phase_{phase}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(mlp, f)

    score = classifier.score(X_test_scaled, y_test)
    return mlp, score, model_path

def load_model(phase=2):
    model_path = os.path.join(MODEL_DIR, f"ann_model_phase_{phase}.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None
