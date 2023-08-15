from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def grid_search(x_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_classifier = RandomForestClassifier()
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_model


def train_model(x_train, y_train, best_model_params):
    model = RandomForestClassifier(**best_model_params, random_state=42)
    model.fit(x_train, y_train)
    return model
