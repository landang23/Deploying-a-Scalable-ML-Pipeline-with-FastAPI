import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import load_model, compute_model_metrics
from train_model import model_path, X_test, X_train, y_test, y_train

def test_one():
    """
    Verify that the trained model is a RandomForestClassifier and has been fitted.

    This test ensures that the expected ML algorithm is being used and that the
    training step was successfully completed by checking for fitted attributes.
    A failure indicates that the wrong model was instantiated or that .fit()
    was never called.
    """

    # Check correct algorithm
    model = load_model(model_path)
    assert isinstance(model, RandomForestClassifier)

    # Check model has been fitted (RandomForest sets estimators_ after fit)
    assert hasattr(model, "estimators_")
    assert len(model.estimators_) > 0


def test_two():
    """
    Validate that evaluation metrics are correctly computed and within valid ranges.

    This test checks that the precision, recall, and F-beta score returned by
    the metrics function are floats and lie between 0 and 1. This helps catch
    implementation errors such as division by zero or incorrect metric logic.
    """

    model = load_model(model_path)
    y_pred = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    # Metrics should be floats
    assert all(isinstance(m, float) for m in [precision, recall, fbeta])

    # Metrics should be in valid range
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_three():
    """
    Ensure consistency and alignment between training and test datasets.

    This test verifies that feature matrices and label vectors have matching
    numbers of samples and that training and test sets have the same number of
    features. It helps prevent subtle bugs caused by misaligned or malformed data.
    """

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
