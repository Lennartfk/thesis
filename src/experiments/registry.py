from src.models.adaptation.baseline import NoAdaptation
from src.models.adaptation.coral import CoralAdapter
from src.models.adaptation.tca import build_tca
from src.models.deep.eegnet import build_eegnet
from src.models.sklearn.logistic_regression import build_logistic_regression
from src.models.sklearn.random_forest import build_random_forest
from src.models.sklearn.svm import build_linear_svm, build_rbf_svm


MODEL_REGISTRY = {
    "svm_linear": build_linear_svm,
    "svm_rbf": build_rbf_svm,
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "eegnet": build_eegnet,
}

ADAPTATION_REGISTRY = {
    "none": NoAdaptation,
    "coral": CoralAdapter,
    "tca": build_tca,
}


def available_models():
    return sorted(MODEL_REGISTRY)


def available_adaptations():
    return sorted(ADAPTATION_REGISTRY)


def get_model(name, seed=42):
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{name}'. Available models: {available_models()}") from exc
    return builder(seed=seed)


def get_adaptation(name):
    try:
        builder = ADAPTATION_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown adaptation '{name}'. Available adaptations: {available_adaptations()}") from exc
    return builder()
