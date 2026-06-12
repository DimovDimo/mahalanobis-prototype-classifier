import numpy as np

class CustomPrototypeClassifier:
    """Baseline Prototype Classifier supporting Euclidean and Mahalanobis spaces."""
    def __init__(self, mode='mahalanobis'):
        if mode not in ['mahalanobis', 'euclidean']:
            raise ValueError("Mode must be either 'mahalanobis' or 'euclidean'")
        self.mode = mode
        self.classes_ = None
        self.prototypes_ = {}
        self.inv_covariances_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            self.prototypes_[c] = np.mean(X_c, axis=0)
            if self.mode == 'mahalanobis':
                cov_matrix = np.cov(X_c, rowvar=False)
                self.inv_covariances_[c] = np.linalg.inv(cov_matrix)
            else:
                self.inv_covariances_[c] = np.eye(X.shape[1])
        return self

    def _compute_distance(self, x, prototype, inv_cov):
        diff = x - prototype
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

    def predict(self, X):
        predictions = []
        for sample in X:
            distances = {c: self._compute_distance(sample, self.prototypes_[c], self.inv_covariances_[c]) for c in self.classes_}
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)


class MultiPrototypeMahalanobisClassifier:
    """Advanced Multimodal Classifier using custom Vector Quantization sub-clustering."""
    def __init__(self, n_prototypes_per_class=2):
        self.n_prototypes = n_prototypes_per_class
        self.classes_ = None
        self.prototypes_ = {}
        self.inv_covariances_ = {}

    def _quantize_sub_clusters(self, X_c):
        sorted_indices = np.argsort(X_c[:, 0])
        X_sorted = X_c[sorted_indices]
        indices = np.linspace(0, len(X_sorted) - 1, self.n_prototypes, dtype=int)
        sub_means = X_sorted[indices].copy()
        for _ in range(10):
            distances = np.sqrt(np.sum((X_c[:, np.newaxis, :] - sub_means[np.newaxis, :, :]) ** 2, axis=2))
            labels = np.argmin(distances, axis=1)
            for m in range(self.n_prototypes):
                assigned_points = X_c[labels == m]
                if len(assigned_points) > 2:
                    sub_means[m] = np.mean(assigned_points, axis=0)
        final_distances = np.sqrt(np.sum((X_c[:, np.newaxis, :] - sub_means[np.newaxis, :, :]) ** 2, axis=2))
        return sub_means, np.argmin(final_distances, axis=1)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            self.prototypes_[c], self.inv_covariances_[c] = [], []
            sub_means, sub_labels = self._quantize_sub_clusters(X_c)
            for m in range(self.n_prototypes):
                X_cm = X_c[sub_labels == m]
                if len(X_cm) < X.shape[1] + 2:
                    X_cm = X_c
                self.prototypes_[c].append(np.mean(X_cm, axis=0))
                cov_matrix = np.cov(X_cm, rowvar=False) + np.eye(X.shape[1]) * 1e-6
                self.inv_covariances_[c].append(np.linalg.inv(cov_matrix))
        return self

    def _compute_mahalanobis(self, x, mu, inv_sigma):
        diff = x - mu
        return np.sqrt(np.dot(np.dot(diff, inv_sigma), diff.T))

    def predict(self, X):
        predictions = []
        for sample in X:
            global_min = float('inf')
            best_class = None
            for c in self.classes_:
                for m in range(self.n_prototypes):
                    dist = self._compute_mahalanobis(sample, self.prototypes_[c][m], self.inv_covariances_[c][m])
                    if dist < global_min:
                        global_min, best_class = dist, c
            predictions.append(best_class)
        return np.array(predictions)


class AdaptiveRobustMPMC(MultiPrototypeMahalanobisClassifier):
    """Production-ready Classifier stabilized via Adaptive Analytic Ridge Shrinkage."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        for c in self.classes_:
            X_c = X[y == c]
            self.prototypes_[c], self.inv_covariances_[c] = [], []
            sub_means, sub_labels = self._quantize_sub_clusters(X_c)
            for m in range(self.n_prototypes):
                X_cm = X_c[sub_labels == m]
                n_samples_cluster = len(X_cm)
                if n_samples_cluster < 2:
                    X_cm = X_c
                    n_samples_cluster = len(X_cm)
                mean_vector = np.mean(X_cm, axis=0)
                raw_cov = np.cov(X_cm, rowvar=False)
                alpha = 1.0 / (1.0 + np.log(n_samples_cluster))
                stabilized_cov = (1.0 - alpha) * raw_cov + alpha * np.eye(n_features)
                self.prototypes_[c].append(mean_vector)
                self.inv_covariances_[c].append(np.linalg.inv(stabilized_cov))
        return self


# 5-LINE DOMAIN-SPECIFIC METRIC ADAPTATIONS
class RedshiftDopplerClassifier(AdaptiveRobustMPMC):
    def fit(self, X, y):
        X_displaced = X.copy(); X_displaced[:, -1] *= 2.5
        return super().fit(X_displaced, y)

class TransitDepthAmplifier(AdaptiveRobustMPMC):
    def _compute_mahalanobis(self, x, mu, inv_sigma):
        amplified_inv = inv_sigma.copy(); amplified_inv *= 5.0
        return np.sqrt(np.dot(np.dot(x - mu, amplified_inv), (x - mu).T))

class CosmicOutlierTrimmedFilter(AdaptiveRobustMPMC):
    def fit(self, X, y):
        robust_X = np.clip(X, np.percentile(X, 5, axis=0), np.percentile(X, 95, axis=0))
        return super().fit(robust_X, y)

class VelocityVarianceEqualizer(AdaptiveRobustMPMC):
    def fit(self, X, y):
        normalized_X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return super().fit(normalized_X, y)

class BackgroundNoiseEraser(AdaptiveRobustMPMC):
    def fit(self, X, y):
        return super().fit(X - np.median(X, axis=0), y)
