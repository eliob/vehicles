

# Convert Categorical values to Ordinal
class ConvertOrdinal():
    def __init__(self, cols):
        self.cols = cols
        self.ordinal_maps = {}

    def fit(self, X, y=None):
        for col in self.cols:
            ordinal_dict = dict(X.groupby(col).apply(lambda grp: grp['price'].median()))
            ordinal_dict = sorted(ordinal_dict.items(), key=lambda kv: kv[1])
            ordinal_dict = dict(ordinal_dict)
            col_map = dict(zip(ordinal_dict.keys(), range(1, len(ordinal_dict) + 1)))
            self.ordinal_maps[col] = col_map
        return self

    def transform(self, X):
        for col in self.cols:
            X[col] = X[col].map(self.ordinal_maps[col])
        return X


# Convert Categorical values to Numerical based on median
class ConvertMedian():
    def __init__(self, cols):
        self.cols = cols
        self.median_maps = {}

    def fit(self, X, y=None):
        for col in self.cols:
            median_dict = dict(X.groupby(col).apply(lambda grp: grp['price'].median()))
            self.median_maps[col] = median_dict
        return self

    def transform(self, X):
        for col in self.cols:
            X[col] = X[col].map(self.median_maps[col])
        return X