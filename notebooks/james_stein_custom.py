# Custom do James Stein para implementar minha lógica de shrinkage para categorias raríssimas
# Esse código vai compor o Lib quando modularizado
import category_encoders as ce
import numpy as np


class _JamesSteinEncoder(ce.JamesSteinEncoder):
    def __init__(self, min_samples: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.min_samples = min_samples

    def _train_independent(self, X, y):
        mapping = {}
        prior = y.mean()
        global_count = len(y)
        global_var = y.var()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            stats = y.groupby(X[col]).agg(['mean', 'var', 'count'])
            i_var = stats['var'].fillna(0)
            unique_cnt = len(X[col].unique())

            smoothing = i_var / (global_var + i_var) * (unique_cnt - 3) / (unique_cnt - 1)
            smoothing = 1 - smoothing
            smoothing = smoothing.clip(lower=0, upper=1)

            # Minha alteração
            if self.min_samples > 0:
                group_counts = stats['count']
                smoothing[group_counts < self.min_samples] = 0.0

            estimate = smoothing * (stats['mean']) + (1 - smoothing) * prior

            if len(stats['mean']) == global_count:
                estimate[:] = prior

            if self.handle_unknown == 'return_nan':
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate.loc[-1] = prior

            if self.handle_missing == 'return_nan':
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate.loc[-2] = prior

            mapping[col] = estimate
        return mapping
