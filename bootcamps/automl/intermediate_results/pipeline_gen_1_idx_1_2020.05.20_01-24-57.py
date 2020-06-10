import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=55)

# Average CV score on the training set was:-693012577.9247557
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.049),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls", max_depth=3, max_features=0.1, min_samples_leaf=7, min_samples_split=10, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
