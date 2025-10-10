from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


def RandomForest():
    return RandomForestClassifier(n_estimators=100, random_state=42)


def LinearSVM():
    return LinearSVC(random_state=42)


def AdaBoost():
    return AdaBoostClassifier(random_state=42)


def GradientBoosting():
    return GradientBoostingClassifier(random_state=42)