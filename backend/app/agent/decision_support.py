"""Stability/cost-aware candidate recommendation and finite stop support."""

from __future__ import annotations

import math
from typing import Any


DECISION_SUPPORT_VERSION = 'agent-decision-support-v1'
VALIDATION_UNCERTAINTY_METHOD = 'row_jackknife_with_group_floor_v1'
MIN_INDEPENDENT_VALIDATION_GROUPS = 10
MIN_INDEPENDENT_GROUPS_PER_CLASS = 3
DEFAULT_UNCERTAINTY_MULTIPLIER = 1.96


def _normalise_confusion_counts(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    size = len(value)
    matrix: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            return None
        normalised_row: list[int] = []
        for count in row:
            try:
                numeric_count = float(count)
            except (OverflowError, TypeError, ValueError):
                return None
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, float))
                or not math.isfinite(numeric_count)
                or numeric_count < 0.0
                or not numeric_count.is_integer()
            ):
                return None
            normalised_row.append(int(count))
        matrix.append(normalised_row)
    return matrix


def _metric_from_confusion_counts(
    matrix: list[list[int]], metric: str
) -> float | None:
    size = len(matrix)
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return None
    row_totals = [sum(row) for row in matrix]
    column_totals = [
        sum(matrix[row][column] for row in range(size))
        for column in range(size)
    ]
    true_positives = [matrix[index][index] for index in range(size)]
    precisions = [
        true_positives[index] / column_totals[index]
        if column_totals[index]
        else 0.0
        for index in range(size)
    ]
    recalls = [
        true_positives[index] / row_totals[index]
        if row_totals[index]
        else 0.0
        for index in range(size)
    ]
    f1_scores = [
        2.0 * precisions[index] * recalls[index]
        / (precisions[index] + recalls[index])
        if precisions[index] + recalls[index]
        else 0.0
        for index in range(size)
    ]
    if metric == 'accuracy':
        return sum(true_positives) / total
    if metric in {'balanced_accuracy', 'macro_recall'}:
        return sum(recalls) / size
    if metric == 'macro_precision':
        return sum(precisions) / size
    if metric == 'macro_f1':
        return sum(f1_scores) / size
    if metric == 'weighted_f1':
        return sum(
            f1_scores[index] * row_totals[index] for index in range(size)
        ) / total
    return None


def estimate_validation_metric_std(
    confusion_counts: Any,
    metric: str,
    *,
    independent_group_count: Any,
    minimum_group_count_per_class: Any,
) -> tuple[float, int] | None:
    """Estimate validation-score standard error without reading Test predictions.

    The row-level delete-one jackknife is bounded below by a Jeffreys-smoothed
    finite-sample proxy based on the independent Validation Sample_ID count.  This
    prevents replicated spectra and perfect small-sample boundaries from claiming
    zero uncertainty.  Invalid or undersized group evidence stays unavailable.
    """

    matrix = _normalise_confusion_counts(confusion_counts)
    if matrix is None:
        return None
    row_count = sum(sum(row) for row in matrix)
    class_count = len(matrix)
    minimum_group_count = max(
        MIN_INDEPENDENT_VALIDATION_GROUPS,
        MIN_INDEPENDENT_GROUPS_PER_CLASS * class_count,
    )
    if (
        isinstance(independent_group_count, bool)
        or not isinstance(independent_group_count, int)
        or independent_group_count < minimum_group_count
        or independent_group_count > row_count
        or isinstance(minimum_group_count_per_class, bool)
        or not isinstance(minimum_group_count_per_class, int)
        or minimum_group_count_per_class < MIN_INDEPENDENT_GROUPS_PER_CLASS
        or minimum_group_count_per_class * class_count > independent_group_count
    ):
        return None
    point_score = _metric_from_confusion_counts(matrix, metric)
    if row_count < 3 or point_score is None:
        return None
    leave_one_values: list[tuple[int, float]] = []
    for row_index, row in enumerate(matrix):
        for column_index, count in enumerate(row):
            if count <= 0:
                continue
            reduced = [list(values) for values in matrix]
            reduced[row_index][column_index] -= 1
            value = _metric_from_confusion_counts(reduced, metric)
            if value is None or not math.isfinite(value):
                return None
            leave_one_values.append((count, value))
    weighted_mean = sum(
        count * value for count, value in leave_one_values
    ) / row_count
    squared_deviation = sum(
        count * (value - weighted_mean) ** 2
        for count, value in leave_one_values
    )
    jackknife_variance = (row_count - 1) / row_count * squared_deviation
    alpha = point_score * independent_group_count + 0.5
    beta = (1.0 - point_score) * independent_group_count + 0.5
    beta_total = alpha + beta
    group_floor_variance = (
        alpha * beta / (beta_total ** 2 * (beta_total + 1.0))
    )
    return (
        math.sqrt(max(jackknife_variance, group_floor_variance)),
        independent_group_count,
    )


def build_decision_support(
    experiments: list[dict[str, Any]],
    *,
    remaining_runs: int,
    proposal_catalog: dict[str, Any] | None,
    target_score: float = 0.90,
    max_generalization_gap: float = 0.05,
    minimum_improvement: float = 0.01,
    uncertainty_multiplier: float = DEFAULT_UNCERTAINTY_MULTIPLIER,
) -> dict[str, Any]:
    if (
        isinstance(uncertainty_multiplier, bool)
        or not isinstance(uncertainty_multiplier, (int, float))
        or not math.isfinite(float(uncertainty_multiplier))
        or float(uncertainty_multiplier) <= 0.0
    ):
        raise ValueError('uncertainty_multiplier must be finite and positive')
    uncertainty_multiplier = float(uncertainty_multiplier)
    assessments: list[dict[str, Any]] = []
    successful_scores: list[tuple[int, float, str]] = []
    used_proposal_ids: set[str] = set()
    failure_count = 0

    for item in experiments:
        action = item.get('effective_action')
        if isinstance(action, dict) and isinstance(action.get('proposal_id'), str):
            used_proposal_ids.add(action['proposal_id'])
        if item.get('state') == 'failed':
            failure_count += 1
        score = item.get('validation_score')
        if (
            item.get('state') != 'succeeded'
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            continue
        diagnosis = item.get('diagnosis')
        gap = (
            diagnosis.get('generalization_gap')
            if isinstance(diagnosis, dict)
            else None
        )
        if isinstance(gap, bool) or not isinstance(gap, (int, float)):
            gap = None
        fits = item.get('model_fit_count')
        if isinstance(fits, bool) or not isinstance(fits, int) or fits < 1:
            fits = None
        validation_std = item.get('validation_std')
        if (
            isinstance(validation_std, bool)
            or not isinstance(validation_std, (int, float))
            or not math.isfinite(float(validation_std))
            or float(validation_std) < 0.0
        ):
            validation_std = None
        conservative_score = (
            float(score) - uncertainty_multiplier * float(validation_std)
            if validation_std is not None
            else float(score)
        )
        gap_penalty = max(0.0, float(gap or 0.0)) * 0.25
        cost_penalty = math.log1p(fits or 1) * 0.01
        utility = round(conservative_score - gap_penalty - cost_penalty, 6)
        assessment = {
            'run_id': str(item['run_id']),
            'attempt': int(item.get('attempt') or 0),
            'validation_score': round(float(score), 6),
            'generalization_gap': (
                round(float(gap), 6) if gap is not None else None
            ),
            'model_fit_count': fits,
            'validation_std': (
                round(float(validation_std), 6)
                if validation_std is not None
                else None
            ),
            'uncertainty_multiplier': (
                uncertainty_multiplier if validation_std is not None else None
            ),
            'conservative_score': round(conservative_score, 6),
            'utility': utility,
        }
        assessments.append(assessment)
        successful_scores.append(
            (assessment['attempt'], assessment['validation_score'], assessment['run_id'])
        )

    assessments.sort(key=lambda item: (-item['utility'], item['attempt']))
    recommended_run_id = assessments[0]['run_id'] if assessments else None
    scores = [item['validation_score'] for item in assessments]
    uncertainty_ready = bool(assessments) and all(
        item['validation_std'] is not None for item in assessments
    )
    uncertainty = {
        'status': 'ready' if uncertainty_ready else 'insufficient_evidence',
        'observation_count': len(scores),
        'candidate_std_available': sum(
            item['validation_std'] is not None for item in assessments
        ),
        'comparative_score_range': (
            round(max(scores) - min(scores), 6) if len(scores) >= 2 else None
        ),
    }

    total_proposals = 0
    catalog_proposal_ids: set[str] = set()
    if isinstance(proposal_catalog, dict):
        proposals = proposal_catalog.get('proposals')
        if isinstance(proposals, list):
            total_proposals = len(proposals)
            catalog_proposal_ids = {
                str(item.get('proposal_id'))
                for item in proposals
                if isinstance(item, dict) and isinstance(item.get('proposal_id'), str)
            }
    unused_proposals = max(
        0,
        total_proposals - len(used_proposal_ids & catalog_proposal_ids),
    )

    action = 'continue'
    reason = 'budget_and_catalog_available'
    if recommended_run_id is None:
        action = 'request_human' if remaining_runs <= 0 else 'continue'
        reason = 'no_successful_candidate'
    else:
        recommended = assessments[0]
        if remaining_runs <= 0:
            action, reason = 'finalize', 'run_budget_exhausted'
        elif total_proposals and unused_proposals == 0:
            action, reason = 'finalize', 'proposal_catalog_exhausted'
        elif (
            uncertainty_ready
            and recommended['conservative_score'] >= target_score
            and recommended['generalization_gap'] is not None
            and recommended['generalization_gap'] <= max_generalization_gap
        ):
            action, reason = 'finalize', 'target_reached'
        else:
            chronological = sorted(successful_scores)
            if uncertainty_ready and len(chronological) >= 3:
                best_so_far = chronological[0][1]
                gains: list[float] = []
                for _, score, _ in chronological[1:]:
                    next_best = max(best_so_far, score)
                    gains.append(next_best - best_so_far)
                    best_so_far = next_best
                if len(gains) >= 2 and max(gains[-2:]) < minimum_improvement:
                    action, reason = 'finalize', 'improvement_plateau'

    return {
        'schema_version': DECISION_SUPPORT_VERSION,
        'status': 'ready' if assessments else 'insufficient_evidence',
        'recommended_run_id': recommended_run_id,
        'candidate_assessments': assessments,
        'uncertainty': uncertainty,
        'failure_count': failure_count,
        'unused_proposal_count': unused_proposals,
        'stop_recommendation': {
            'action': action,
            'reason': reason,
        },
    }
