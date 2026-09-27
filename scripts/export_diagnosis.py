"""Read-only canonical diagnostic export. No HTTP, migration or model loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_poc.orchestration.runtime import read_status, _journal_path_digest
from backend.app.agent.diagnosis import DiagnosisReport, DiagnosisInput, digest
from scripts.task_cost_report import readonly_snapshot, _journal_budget, call_rows
from agent_poc.orchestration.diagnosis import call_cost_evidence, verify_report_proposal


def build_diagnosis(*, storage: Path, thread_id: str) -> dict:
    state=read_status(storage=storage,thread_id=thread_id)
    result=dict(schema_version='agent-diagnosis-export-v1',freshness='snapshot_only',
        status='disabled',reason_code='diagnosis_not_provided',report=None,
        current_eligibility='unverified',cost_evidence=None)
    if state['versions']['state']!='agent-state-v9':
        return result
    if state['task']['canonical_journal_sha256']!=_journal_path_digest(storage):
        raise ValueError('diagnosis_canonical_storage_mismatch')
    if not state['task']['diagnosis_policy']['enabled']:
        return result
    journal=storage/'calls.sqlite'
    task=state['identity']['task_id']
    balance=_journal_budget(journal,thread_id=thread_id,task_id=task,
        policy_digest=state['task']['budget_policy_digest'])
    if balance is None:
        raise ValueError('diagnosis_journal_binding_missing')
    db=readonly_snapshot(journal)
    try:
        db.execute('BEGIN')
        rows=db.execute('SELECT payload_json,payload_digest FROM diagnosis_reports_v1 WHERE thread_id=? ORDER BY rowid',
                        (thread_id,)).fetchall()
        if len(rows)>2:
            raise ValueError('diagnosis_report_limit_exceeded')
        reports=[]
        training_cost=None
        for row in rows:
            report=DiagnosisReport.model_validate_json(row['payload_json']).model_dump(mode='json')
            if digest(report)!=row['payload_digest'] or report['bindings']['task']!=digest(task):
                raise ValueError('diagnosis_report_integrity_failed')
            raw=db.execute('SELECT payload_json,payload_digest FROM diagnosis_inputs_v1 WHERE thread_id=? AND input_digest=?',
                (thread_id,report['input_digest'])).fetchone()
            if raw is None:
                raise ValueError('diagnosis_input_missing')
            snapshot=DiagnosisInput.model_validate_json(raw['payload_json']).model_dump(mode='json')
            if (digest(snapshot)!=raw['payload_digest'] or report['bindings']!=snapshot['bindings']
                    or digest(snapshot['context'])!=snapshot['displayed_context_digest']):
                raise ValueError('diagnosis_input_integrity_failed')
            DiagnosisReport.model_validate(report).bind(snapshot)
            verify_report_proposal(db,thread_id,report,snapshot)
            if snapshot['context']['context_version']=='agent-context-diagnosis-unavailable-v1':
                result['capacity_receipt']=snapshot['context']
            report.setdefault('assessment','historically_unrecorded')
            report['freshness']='snapshot_only'
            reports.append(report)
            training_cost=snapshot['context'].get('training_cost')
        if reports:
            result.update(status=reports[-1]['status'],reason_code=reports[-1]['reason_code'],report=reports[-1])
        else:
            result.update(status='unavailable',reason_code='diagnosis_not_provided')
        result['cost_evidence']=dict(schema_version='diagnosis-cost-evidence-v1',
            artifact_status='unverified',training=training_cost,
            calls=call_cost_evidence(balance,call_rows(journal,thread_id)[0],task),
            source_ref='journal-'+digest(task),limitation='Ledger evidence does not certify Run artifacts.')
    finally:
        db.close()
    return result


def export_diagnosis(*, storage: Path, thread_id: str, output: Path):
    payload=build_diagnosis(storage=storage,thread_id=thread_id)
    # Refuse an existing destination, including empty directories: exports never overwrite evidence.
    output.mkdir(parents=True,exist_ok=False)
    (output/'diagnosis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Feedback diagnosis', '', 'Historical snapshot only; current eligibility is unverified.',
        '', f"Status: {payload['status']}", f"Reason: {payload['reason_code'] or 'none'}", '']
    report=payload['report']
    if report:
        lines += [f"Assessment: {report['assessment'] or 'not_applicable'}", '', '## Verified facts','']
        lines += [f"- {f['fact_id']}: {f['code']} = {json.dumps(f['value'],ensure_ascii=False)} ({f['trust']}; source {f['source_ref']})" for f in report['facts']]
        lines+=['','## Hypotheses (not confirmed causes)','']
        lines += [f"- {h['explanation']} Uncertainty: {h['uncertainty']}. Evidence: {', '.join(h['evidence_refs'])}." for h in report['hypotheses']]
        lines+=['','## Suggestions (execution disabled)','']
        lines += [f"- {s['action_id']}: {s['rationale']} Conditions: {', '.join(s['condition_refs']) or 'none'}." for s in report['suggestions']]
        lines+=['','## Limitations','']+[f'- {code}' for code in report['limitations']]
        lines+=['',f"Report: {report['diagnosis_id']}",f"Input: {report['input_digest']}",
            f"Rules: {report['rules_version']}",'']
    (output/'diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage',type=Path,required=True)
    parser.add_argument('--thread-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    export_diagnosis(storage=args.storage,thread_id=args.thread_id,output=args.output)


if __name__=='__main__':
    main()
