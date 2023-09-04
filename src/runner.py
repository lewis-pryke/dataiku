from pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

LABEL = 'target'
SEED = 42
MODELS = {
        'RandomForest': RandomForestClassifier(random_state=SEED),
        'GradientBoosting': GradientBoostingClassifier(random_state=SEED)
        }

pipeline = Pipeline(label_col=LABEL, seed=SEED, models=MODELS)
pipeline.run()
