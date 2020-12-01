from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential

from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.rules import VeryFastDecisionRulesClassifier
from skmultiflow.meta import AdditiveExpertEnsembleClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier

PATH = './airlines.csv'


def flow_detection_classifier(classifier, stream):
    evaluator = EvaluatePrequential(show_plot=True, pretrain_size=2000, max_samples=50000)
    evaluator.evaluate(stream=stream, model=classifier)
    return evaluator


def make_stream(path, classifier):
    stream = FileStream(path)
    evaluator = flow_detection_classifier(classifier, stream)
    stream = evaluator.stream.y
    return stream


# Streams based on classifiers:
# KNNClassifier
make_stream(PATH, KNNClassifier(n_neighbors=5, max_window_size=1000, leaf_size=30))
make_stream(PATH, KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40))

# HoeffdingTreeClassifier
make_stream(PATH, HoeffdingTreeClassifier(memory_estimate_period=1000000, grace_period=200, leaf_prediction='nba'))
make_stream(PATH, HoeffdingTreeClassifier(memory_estimate_period=2000000, grace_period=300, leaf_prediction='mc'))

# AdditiveExpertEnsembleClassifier
make_stream(PATH, AdditiveExpertEnsembleClassifier(n_estimators=5, beta=0.8, gamma=0.1, pruning='weakest'))
make_stream(PATH, AdditiveExpertEnsembleClassifier(n_estimators=8, beta=0.9, gamma=0.3, pruning='oldest'))

# VeryFastDecisionRulesClassifier
make_stream(PATH, VeryFastDecisionRulesClassifier(grace_period=200, tie_threshold=0.05, max_rules=20))
make_stream(PATH, VeryFastDecisionRulesClassifier(grace_period=300, tie_threshold=0.1, max_rules=30))

# AdaptiveRandomForestClassifier
make_stream(PATH, AdaptiveRandomForestClassifier(n_estimators=10, lambda_value=6, performance_metric='acc'))
make_stream(PATH, AdaptiveRandomForestClassifier(n_estimators=20, lambda_value=8, performance_metric='kappa'))

# AccuracyWeightedEnsembleClassifier
make_stream(PATH, AccuracyWeightedEnsembleClassifier(n_estimators=10, n_kept_estimators=30, window_size=200, n_splits=5))
make_stream(PATH, AccuracyWeightedEnsembleClassifier(n_estimators=15, n_kept_estimators=40, window_size=300, n_splits=8))
