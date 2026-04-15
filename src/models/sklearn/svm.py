from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def build_linear_svm(seed=42):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(random_state=seed, max_iter=20000)),
        ]
    )


def build_rbf_svm(seed=42):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", random_state=seed)),
        ]
    )
