class NoAdaptation:
    name = "none"
    uses_target_data = False

    def fit(self, X_train, y_train=None, X_target_unlabeled=None):
        return self

    def transform_train(self, X_train):
        return X_train

    def transform_test(self, X_test):
        return X_test

    def fit_transform(self, X_train, X_test, y_train=None):
        self.fit(X_train, y_train=y_train, X_target_unlabeled=None)
        return self.transform_train(X_train), self.transform_test(X_test)
