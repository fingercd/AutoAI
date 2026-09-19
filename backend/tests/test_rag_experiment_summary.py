from scripts.agent_ablation import summarize_plan


def test_equal_run_means_pairs_and_failed_denominators():
    rows = []
    records = {}
    for task in range(10):
        for family in ('ML', 'DL'):
            for side in ('on', 'off'):
                identity = f'{task}-{family}-{side}'
                rows.append(dict(task_id=str(task), family=family, knowledge=side,
                                 seed=42, repeat=0, experiment_id=identity))
                score = (.8 if family == 'ML' else .6) + (.1 if side == 'on' else 0)
                records[identity] = dict(status='completed', run_id='run_'+identity,
                    offline_test={key:score for key in ('macro_f1', 'accuracy', 'balanced_accuracy')})
    summary = summarize_plan(rows, records)
    assert abs(summary['groups']['on']['metrics']['macro_f1']['all']['complete_mean'] - .8) < 1e-12
    assert abs(summary['paired_mean_difference']['macro_f1'] - .1) < 1e-12
    records['0-ML-on'] = dict(status='failed', failure_reason='training_failed')
    summary = summarize_plan(rows, records)
    on = summary['groups']['on']
    assert len(on['rows']) == on['planned'] == 20 and on['succeeded'] == 19
    assert on['metrics']['macro_f1']['all']['complete_mean'] is None
    assert on['metrics']['macro_f1']['all']['denominator'] == 19
    assert summary['paired_count'] == 19 and summary['unpaired_count'] == 1
