"""One evaluation unit for the approved diagnostic Valid and Test partitions."""
from collections import defaultdict
import math

EVALUATION_VERSION = 'sample-id-mean-probability-v1'

def grouped_probability_predictions(rows, labels, expected_groups=None):
    labels = [str(label) for label in labels]
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['Sample_ID'])].append(row)
    if expected_groups is not None and set(groups) != set(map(str, expected_groups)):
        raise ValueError('prediction Sample_ID coverage differs from frozen partition')
    result = []
    for sample_id in sorted(groups):
        observations = groups[sample_id]
        truths = {str(row['true_label']) for row in observations}
        if len(truths) != 1 or not truths <= set(labels):
            raise ValueError('prediction group has inconsistent or unknown true labels')
        probabilities = {}
        for label in labels:
            values = [float(row['prob_' + label]) for row in observations]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError('invalid class probability')
            probabilities[label] = sum(values) / len(values)
        if abs(sum(probabilities.values()) - 1) > 1e-5:
            raise ValueError('class probabilities do not sum to one')
        # Python max preserves the frozen class order for exact ties, as v4 Test did.
        predicted = max(labels, key=probabilities.__getitem__)
        result.append(dict(sample_id=sample_id, true_label=next(iter(truths)),
                           predicted_label=predicted, probabilities=probabilities))
    return result

def grouped_balanced_accuracy(rows, labels, expected_groups=None):
    predictions = grouped_probability_predictions(rows, labels, expected_groups)
    recalls = []
    for label in map(str, labels):
        actual = [row for row in predictions if row['true_label'] == label]
        if not actual:
            raise ValueError('evaluation partition is not class complete')
        recalls.append(sum(row['predicted_label'] == label for row in actual) / len(actual))
    return sum(recalls) / len(recalls), predictions


def grouped_evaluation_payload(evaluation, sample_ids, labels):
    """Convert row probabilities to the same Sample_ID prediction used for Test.

    Encoded true/pred arrays in the returned payload describe groups. The input
    row payload stays intact for row-level audit and predictions.csv.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
        confusion_matrix, f1_score, precision_score, recall_score)
    labels = list(map(str, labels))
    rows = [dict(Sample_ID=str(group), true_label=labels[int(truth)],
                 **{'prob_' + label: float(value) for label, value in zip(labels, probabilities, strict=True)})
            for group, truth, probabilities in zip(sample_ids, evaluation['true'],
                                                   evaluation['probabilities'], strict=True)]
    predictions = grouped_probability_predictions(rows, labels, set(map(str, sample_ids)))
    label_ids = {label: i for i, label in enumerate(labels)}
    true = [label_ids[row['true_label']] for row in predictions]
    pred = [label_ids[row['predicted_label']] for row in predictions]
    return dict(evaluation_version=EVALUATION_VERSION, sample_ids=[row['sample_id'] for row in predictions],
        true=true, pred=pred, probabilities=[[row['probabilities'][label] for label in labels] for row in predictions],
        accuracy=float(accuracy_score(true, pred)), balanced_accuracy=float(balanced_accuracy_score(true, pred)),
        macro_f1=float(f1_score(true, pred, average='macro', zero_division=0)),
        weighted_f1=float(f1_score(true, pred, average='weighted', zero_division=0)),
        precision=float(precision_score(true, pred, average='macro', zero_division=0)),
        recall=float(recall_score(true, pred, average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(true, pred, labels=list(range(len(labels)))).tolist())
