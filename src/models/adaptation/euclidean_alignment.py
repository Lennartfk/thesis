import numpy as np


class EuclideanAlignment:
    """Subject-wise Euclidean Alignment for EEG epochs.

    For each subject, estimate the mean spatial covariance over unlabeled epochs and
    left-multiply every epoch by the inverse square root of that covariance.
    """

    name = "ea"
    uses_target_data = True

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.transforms_ = {}

    def _mean_covariance(self, X):
        covariances = np.matmul(X, np.swapaxes(X, 1, 2)) / X.shape[2]
        covariance = covariances.mean(axis=0)
        covariance = (covariance + covariance.T) / 2.0
        covariance += self.eps * np.eye(covariance.shape[0], dtype=covariance.dtype)
        return covariance

    def _inverse_square_root(self, covariance):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, self.eps)
        inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        return inverse_sqrt.astype(np.float32, copy=False)

    def fit(self, X, subject_ids):
        subject_ids = np.asarray(subject_ids).astype(str)
        self.transforms_ = {}
        for subject_id in sorted(np.unique(subject_ids)):
            subject_X = X[subject_ids == subject_id]
            covariance = self._mean_covariance(subject_X)
            self.transforms_[subject_id] = self._inverse_square_root(covariance)
        return self

    def transform(self, X, subject_ids):
        subject_ids = np.asarray(subject_ids).astype(str)
        aligned = np.empty_like(X, dtype=np.float32)
        for subject_id in np.unique(subject_ids):
            if subject_id not in self.transforms_:
                raise KeyError(f"No Euclidean Alignment transform was fit for subject {subject_id}.")
            subject_mask = subject_ids == subject_id
            aligned[subject_mask] = np.einsum("ct,nts->ncs", self.transforms_[subject_id], X[subject_mask])
        return aligned

    def fit_transform(self, X, subject_ids):
        self.fit(X, subject_ids)
        return self.transform(X, subject_ids)


def align_subjects(X, subject_ids, eps=1e-6):
    aligner = EuclideanAlignment(eps=eps)
    aligned = aligner.fit_transform(X, subject_ids)
    return aligned, aligner
