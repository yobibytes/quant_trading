import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=55)

# Average CV score on the training set was:-639264539.9247355
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.001, loss="ls", max_depth=10, max_features=0.05, min_samples_leaf=17, min_samples_split=2, n_estimators=100, subsample=0.55)),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls", max_depth=3, max_features=0.1, min_samples_leaf=7, min_samples_split=10, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
