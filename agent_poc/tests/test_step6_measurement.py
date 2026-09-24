"""Request-level timing remains durable and never turns unknown tokens into zero."""
from __future__ import annotations

from agent_poc.orchestration.persistence import CallJournal


def test_llm_request_export_survives_retry_and_restart(tmp_path):
    path = tmp_path / 'calls.sqlite3'
    journal = CallJournal(path, 'task-1')
    first = journal.begin(operation_id='choose', kind='llm', name='selection',
                          maximum=3, max_attempts=3, deadline=10**12)
    journal.finish(first, error_code='provider_timeout', measurement=dict(
        request_started_at_utc='2026-09-24T01:00:00+00:00',
        response_ended_at_utc='2026-09-24T01:00:02+00:00',
        request_duration_seconds=2.0, phase='selection', protocol='json_action',
        model_id_sha256='a' * 64, model_id='fixture', outcome='failed'))
    second = journal.begin(operation_id='choose', kind='llm', name='selection',
                           maximum=3, max_attempts=3, deadline=10**12)
    journal.finish(second, input_tokens=19, output_tokens=3, total_tokens=22,
                   token_status='known', measurement=dict(
        request_started_at_utc='2026-09-24T01:00:03+00:00',
        response_ended_at_utc='2026-09-24T01:00:04+00:00',
        request_duration_seconds=1.0, phase='selection', protocol='json_action',
        model_id_sha256='a' * 64, model_id='fixture', outcome='response_received'))
    journal.begin_monitor_wait(operation_id='poll-1',run_id='run-1',reason_code='queued')
    journal.begin_monitor_wait(operation_id='poll-1',run_id='run-1',reason_code='queued')
    journal.end_monitor_wait(operation_id='poll-1')
    journal.end_monitor_wait(operation_id='poll-1')
    journal.close()

    reopened = CallJournal(path, 'task-1')
    exported = reopened.export_llm_requests()
    assert [(row['retry_index'], row['status']) for row in exported] == [
        (0, 'failed'), (1, 'confirmed')]
    assert exported[0]['input_tokens'] is None
    assert exported[0]['total_tokens'] is None
    assert exported[0]['token_status'] == 'unknown'
    assert exported[1]['total_tokens'] == 22
    assert exported[1]['model_id'] == 'fixture'
    assert exported[1]['request_duration_seconds'] == 1.0
    waits=reopened.export_monitor_waits()
    assert len(waits)==1 and waits[0]['status']=='succeeded'
    assert waits[0]['duration_seconds'] is not None
    reopened.close()
