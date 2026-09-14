"""Bounded LangGraph ticks over the six public Agent tools.

Each tick persists a prepared operation before its side effect, then ends.
The runner wakes waiting ticks; there is no polling recursion inside the graph.
Training remains exclusively behind the HTTP queue and independent worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Callable

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agent_poc.clients.autoai_client import (
    AutoAIClient, AgentClientError, AgentHTTPError, AgentContractError,
    AgentConnectionError, AgentTimeoutError,
)
from agent_poc.tools import ToolDispatcher
from .llm import LLMAdapter, LLMError
from .persistence import CallJournal, PersistenceError
from .projection import selection_context, finalization_context, recipe_selection_context
from .state import (
    GraphState, StateModel, SessionRequest, ExperimentRequest, FinalizeRequest,
    Candidate, DecisionState, HistoryEvent, UsageRecord, OperationAttempts,
    apply_patch, fingerprint, validate_state,
)


@dataclass(repr=False)
class Dependencies:
    client: AutoAIClient
    llm: LLMAdapter
    journal: CallJournal
    clock: Callable[[], float] = time.time


TOOLS = {
    'capabilities': 'inspect_ml_capabilities', 'session': 'start_ml_session',
    'inspect_session': 'inspect_ml_session', 'choose': 'submit_ml_experiment',
    'submit': 'submit_ml_experiment', 'observe': 'observe_ml_experiment',
    'finalize_decision': 'finalize_ml_session', 'finalize': 'finalize_ml_session',
    'confirm': 'inspect_ml_session', 'reconcile': None,
}


def _key(prefix, value, suffix=''):
    result = f'{prefix}-{value}{suffix}'
    return result if len(result) <= 120 else f'{prefix}-{fingerprint(value)[:32]}{suffix}'


def _next(state, action, *, status='running', reason=None, wake=None):
    return apply_patch(state, {
        'lifecycle': {'status': status, 'stage': action or state['lifecycle']['stage'],
                      'next_action': action, 'reason_code': reason},
        'recovery': {'next_wake_at': wake, 'reason_code': reason},
    })


def _stop(state, reason, now, *, status='failed', uncertain=False):
    patch = {
        'lifecycle': {'status': status, 'next_action': None, 'ended_at': now, 'reason_code': reason},
        'recovery': {'needs_human_review': uncertain, 'reason_code': reason, 'next_wake_at': None},
        'finalization': {'termination_reason': reason},
    }
    if not state['candidates']['items']:
        patch['finalization'].update(status='unselected', selected_run_id=None)
        patch['candidates'] = {'status': 'invalid'}
    return apply_patch(state, patch)


class Nodes:
    def __init__(self, dependencies):
        self.deps = dependencies
        if dependencies.client.max_retries != 0:
            raise ValueError('graph requires client max_retries=0 for durable accounting')
        self.dispatcher = ToolDispatcher(dependencies.client)

    def account(self, state):
        rows = self.deps.journal.snapshot()
        budget = {'usage': []}
        attempts = {}
        for row in rows:
            unknown = row['status'] == 'dispatched'
            tokens_known = row['input_tokens'] is not None and row['output_tokens'] is not None
            usage = UsageRecord(
                usage_id=f"call-{row['id']}", operation_id=row['operation_id'], kind=row['kind'],
                status='unknown' if unknown else 'confirmed', input_tokens=row['input_tokens'],
                output_tokens=row['output_tokens'],
                total_tokens=row['total_tokens'],
                token_status=('not_applicable' if row['kind'] == 'api' else
                              row['token_status'] or ('known' if tokens_known else 'unknown')),
            ).model_dump(mode='json')
            budget['usage'].append(usage)
            entry = attempts.setdefault(row['operation_id'], OperationAttempts(
                operation_id=row['operation_id']).model_dump(mode='json'))
            entry['network_attempts'] += 1
            if row['name'] == 'reconcile_ml_session':
                entry['reconciliation_attempts'] += 1
            if row['kind'] == 'llm' and row['error_code']:
                entry['repair_attempts'] += 1
            entry['last_error_code'] = row['error_code']
        for kind in ('api', 'llm'):
            selected = [row for row in rows if row['kind'] == kind]
            limit = state['budget'][kind + '_calls']['limit']
            budget[kind + '_calls'] = {'actual': float(len(selected)), 'reserved': 0.0,
                'remaining': max(0.0, limit - len(selected)),
                'unknown_pending': sum(row['status'] == 'dispatched' for row in selected),
                'measurement_status': 'ready'}
        llm_rows = [row for row in rows if row['kind'] == 'llm']
        budget['cached_tokens'] = {'unknown_pending':len(llm_rows), 'measurement_status':'unavailable'}
        for dimension in ('input_tokens', 'output_tokens'):
            known = [row[dimension] for row in llm_rows if row[dimension] is not None]
            unknown = sum(row[dimension] is None for row in llm_rows)
            budget[dimension] = {'actual': float(sum(known)) if known else None,
                                 'unknown_pending': unknown,
                                 'measurement_status': 'pending' if unknown or not llm_rows else 'ready'}
        budget['wall_time'] = {'actual': max(0.0, self.deps.clock() - state['lifecycle']['started_at']),
                              'remaining': max(0.0, state['budget']['deadline_at'] - self.deps.clock()),
                              'measurement_status': 'ready'}
        return apply_patch(state, {'budget': budget, 'recovery': {'attempts': list(attempts.values())}})

    @staticmethod
    def preparation(state):
        if state['versions']['state']!='agent-state-v3' or state['evidence']['content'] is None:
            return None
        return dict(evaluation_plan=state['recipes']['evaluation_plan'],
                    evidence=state['evidence']['content'],catalog=state['recipes']['catalog'])

    def restore_client_contract(self, state):
        """Rehydrate only checkpointed facts, including when prepare is skipped."""
        if state['versions']['api'] == 'agent-session-v2':
            from agent_poc.tools import build_tool_schemas
            source = state['capabilities']['frozen_snapshot'] or state['capabilities']['wire_snapshot']
            if source is not None:
                health = {k:v for k,v in source.items() if k != 'model_configs'}
                health.update(contract_version='agent-session-v2',status='ready',modules={},capabilities=dict(create_session=True,create_experiment=True,read_session=True,read_feedback=True,finalize_session=True))
                self.deps.client.tool_schemas = build_tool_schemas(health)
                if state['versions']['state']=='agent-state-v3':
                    self.deps.client._recipe_tools()
            frozen = state['capabilities']['frozen_snapshot']
            if frozen is not None and state['identity']['session_id'] is not None:
                self.deps.client.restore_frozen_session(state['identity']['session_id'], frozen, **({'preparation':self.preparation(state)} if state['versions']['state']=='agent-state-v3' else {}))

    def prepare(self, raw):
        state = validate_state(raw).model_dump(mode='json')
        self.restore_client_contract(state)
        state = self.account(state)
        action = state['lifecycle']['next_action']
        if action is None:
            return state
        if self.deps.clock() >= state['budget']['deadline_at']:
            return _stop(state, 'deadline_exceeded', self.deps.clock(), status='timed_out',
                         uncertain=state['execution']['run_status'] in ('queued', 'running')
                         or state['recovery']['pending_operation'] is not None)
        if action not in TOOLS:
            return _stop(state, 'unknown_graph_action', self.deps.clock())
        wake = state['recovery']['next_wake_at']
        if wake is not None and self.deps.clock() < wake:
            return state
        identity, task = state['identity'], state['task']
        identity_patch = {}
        content = None
        request_id = None
        kind = {'session':'session', 'submit':'experiment', 'finalize':'finalize',
                'reconcile':'reconcile', 'observe':'observe', 'choose':'llm',
                'finalize_decision':'llm'}.get(action, 'inspect')
        op_id = _key(action, identity['task_id'], f"-{len(state['history']['events'])}")
        if action == 'session':
            op_id = identity['session_operation_id'] or _key('session', identity['task_id'])
            request_id = identity['session_request_id'] or op_id
            identity_patch.update(session_operation_id=op_id, session_request_id=request_id)
            request_type = SessionRequest
            request_extra = {}
            if state['versions']['api'] == 'agent-session-v2':
                from .state_v2 import SessionRequest as request_type
                if state['versions']['state']=='agent-state-v3':
                    from .state import RecipeSessionRequest as request_type
                request_extra['model_configs'] = {k:v for k,v in task['model_configs'].items() if k in state['capabilities']['eligible_models']}
            content = request_type(**request_extra, dataset_id=task['dataset_id'], selection_metric=task['selection_metric'],
                allowed_models=state['capabilities']['eligible_models'], seed=task['seed'],
                client_request_id=request_id, context_policy=state['module_policy']['context_policy']).model_dump(mode='json')
        elif action == 'choose':
            experiment_id = identity['current_experiment_id'] or _key('experiment', identity['task_id'])
            request_id = identity['experiment_request_id'] or experiment_id
            identity_patch.update(current_experiment_id=experiment_id,
                experiment_operation_id=experiment_id, experiment_request_id=request_id)
            op_id = _key('choose', identity['task_id'])
        elif action == 'submit':
            op_id = identity['experiment_operation_id']
            request_id = identity['experiment_request_id']
            content = state['execution']['submission_content']
        elif action == 'finalize_decision':
            op_id = _key('finalize-decision', identity['task_id'])
        elif action == 'finalize':
            op_id = identity['finalize_operation_id'] or _key('finalize', identity['task_id'])
            identity_patch['finalize_operation_id'] = op_id
            content = FinalizeRequest(session_id=identity['session_id'],
                selected_run_id=state['finalization']['selected_run_id']).model_dump(mode='json')
        elif action == 'reconcile':
            op_id = _key('reconcile', identity['task_id'])
        previous = state['recovery']['pending_operation']
        if (previous and previous['status'] in ('failed', 'unknown')
                and state['lifecycle']['stage'] == action and previous['kind'] == kind):
            op_id = previous['operation_id']
        # Replaying a prepared side effect uses the same immutable request body.
        pending = {'operation_id': op_id, 'kind': kind, 'request_id': request_id,
                   'tool_name': TOOLS[action], 'content': content,
                   'content_fingerprint': fingerprint(content) if content else None, 'status': 'prepared'}
        return apply_patch(state, {'identity': identity_patch, 'recovery': {
            'pending_operation': pending, 'next_wake_at': None},
            'lifecycle': {'status':'running', 'stage':action}})

    def call(self, state, name, arguments, *, suffix=''):
        self.restore_client_contract(validate_state(state).model_dump(mode='json'))
        pending = state['recovery']['pending_operation']
        op_id = pending['operation_id'] + suffix
        call_id = self.deps.journal.begin(operation_id=op_id, kind='api', name=name,
            maximum=int(state['budget']['api_calls']['limit']),
            max_attempts=state['budget']['max_operation_attempts'],
            deadline=state['budget']['deadline_at'], now=self.deps.clock())
        original_timeout = getattr(self.deps.client, 'timeout', None)
        if original_timeout is not None:
            self.deps.client.timeout = min(original_timeout, state['budget']['deadline_at'] - self.deps.clock())
        try:
            if name == 'reconcile_ml_session':
                result = self.deps.client.reconcile_ml_session(arguments['session_id'])
            else:
                result = self.dispatcher.dispatch(name, arguments)
        except AgentClientError as error:
            code = (error.code if isinstance(error, AgentHTTPError) else
                    'api_contract_invalid' if isinstance(error, AgentContractError) else 'api_unavailable')
            self.deps.journal.finish(call_id, error_code=code)
            raise
        finally:
            if original_timeout is not None:
                self.deps.client.timeout = original_timeout
        self.deps.journal.finish(call_id)
        return result

    def llm_call(self, state, phase, context):
        call_id = self.deps.journal.begin(
            operation_id=state['recovery']['pending_operation']['operation_id'], kind='llm', name=phase,
            maximum=int(state['budget']['llm_calls']['limit']),
            max_attempts=state['budget']['max_repair_attempts'] + 1,
            deadline=state['budget']['deadline_at'], now=self.deps.clock())
        try:
            if isinstance(self.deps.llm, LLMAdapter):
                operation_id = state['recovery']['pending_operation']['operation_id']
                repair_code = next((entry['last_error_code'] for entry in state['recovery']['attempts']
                                    if entry['operation_id'] == operation_id), None)
                proposal = self.deps.llm.propose(phase, context,
                    timeout_seconds=state['budget']['deadline_at'] - self.deps.clock(),
                    repair_code=repair_code)
            else:
                proposal = self.deps.llm.propose(phase, context)
        except LLMError as error:
            self.deps.journal.finish(call_id, error_code=error.code,
                input_tokens=error.usage.prompt_tokens, output_tokens=error.usage.completion_tokens,
                total_tokens=error.usage.total_tokens, token_status=error.usage.status)
            raise
        self.deps.journal.finish(call_id, input_tokens=proposal.usage.prompt_tokens,
                                 output_tokens=proposal.usage.completion_tokens,
                                 total_tokens=proposal.usage.total_tokens, token_status=proposal.usage.status)
        return proposal

    def execute(self, action, raw):
        state = validate_state(raw).model_dump(mode='json')
        operation_id = state['recovery']['pending_operation']['operation_id']
        confirmed = False
        try:
            state = getattr(self, action)(state)
            confirmed = True
        except LLMError as error:
            state = self.account(state)
            op = state['recovery']['pending_operation']['operation_id']
            failures = next((x['repair_attempts'] for x in state['recovery']['attempts']
                             if x['operation_id'] == op), 0)
            if failures > state['budget']['max_repair_attempts']:
                state = _stop(state, 'llm_repair_exhausted', self.deps.clock())
            else:
                state = _next(state, action, status='waiting', reason=error.code, wake=self.deps.clock()+1)
        except AgentHTTPError as error:
            if error.code == 'agent_request_released':
                state = _stop(state, error.code, self.deps.clock())
            elif error.code in ('agent_compensation_required', 'agent_active_run_exists'):
                state = _next(state, 'reconcile', status='recovering', reason=error.code)
            elif error.retryable:
                state = self.retry(state, action, error.code)
            else:
                state = _stop(state, error.code or 'api_rejected', self.deps.clock(),
                              status='needs_attention', uncertain=True)
        except (AgentConnectionError, AgentTimeoutError):
            state = self.retry(state, action, 'api_unavailable')
        except PersistenceError as error:
            code = str(error)
            state = _stop(state, code, self.deps.clock(),
                status='timed_out' if code == 'deadline_exceeded' else 'needs_attention', uncertain=True)
        except (AgentContractError, ValidationError, ValueError, KeyError, TypeError):
            state = _stop(state, 'contract_validation_failed', self.deps.clock(),
                          status='needs_attention', uncertain=True)
        # Never let raw provider/client exceptions enter checkpoint task errors.
        except Exception:
            state = _stop(state, 'orchestration_internal_error', self.deps.clock(),
                          status='needs_attention', uncertain=True)
        if state['recovery']['pending_operation'] is not None:
            state = apply_patch(state, {'recovery': {'pending_operation': {
                **state['recovery']['pending_operation'],
                'status':'confirmed' if confirmed else 'unknown'}}})
        state = self.account(state)
        rows = self.deps.journal.snapshot()
        if rows:
            event_id = f"event-call-{rows[-1]['id']}"
            if not any(event['event_id'] == event_id for event in state['history']['events']):
                event = HistoryEvent(event_id=event_id, kind=action,
                    sequence=len(state['history']['events']), occurred_at=self.deps.clock(),
                    operation_id=operation_id,
                    decision_id=state['decision']['decision_id'], run_id=state['execution']['run_id'],
                    experiment_id=state['execution']['experiment_id'],
                    reason_code=state['lifecycle']['reason_code'],
                    safe_record_ref=f"call-{rows[-1]['id']}").model_dump(mode='json')
                state = apply_patch(state, {'history': {'events': [event]}})
        return state

    def retry(self, state, action, reason):
        target = {'submit':'inspect_session', 'finalize':'confirm'}.get(action, action)
        return _next(state, target, status='waiting', reason=reason, wake=self.deps.clock()+2)

    def remaining(self, state, response):
        remaining = response['remaining_runs']
        if remaining not in (0, 1):
            raise ValueError('backend budget violates protocol')
        return apply_patch(state, {'budget': {'runs': {'remaining': float(remaining),
            'actual': float(1-remaining) if state['execution']['run_id'] else None,
            'reserved': float(1-remaining) if not state['execution']['run_id'] else 0.0,
            'measurement_status':'ready'}}, 'recovery': {'last_confirmed_backend_state': {
                'remaining_runs':remaining, 'confirmed_at':self.deps.clock()}}})

    def capabilities(self, state):
        response = self.call(state, 'inspect_ml_capabilities', {})
        step2 = state['versions']['api'] == 'agent-session-v2'
        if step2:
            from agent_poc.clients.contracts_v2 import ModelCapability, validate_params
            declared = {m['id']: ModelCapability.model_validate(m) for m in response['models']}
            if set(state['task']['allowed_models']) - set(declared):
                raise ValueError('unknown allowed model')
            if set(state['task']['model_configs']) - set(state['task']['allowed_models']):
                raise ValueError('model_configs keys must belong to allowed_models')
            for name, params in state['task']['model_configs'].items():
                validate_params(declared[name], params)
        models = [m['id'] for m in response['models'] if m['available']] if step2 else response['models']
        eligible = [name for name in state['task']['allowed_models'] if name in models]
        if not eligible or not response['capabilities']['create_experiment']:
            return _stop(state, 'no_available_model', self.deps.clock())
        snapshot = {'status':'ready', 'observed_at':self.deps.clock(),
            'models':[{'model_type':name, 'available':True} for name in models],
            'eligible_models':eligible,
            'tools':[{'name':name, 'available':available, 'status':'ready' if available else 'unavailable'}
                     for name,available in response['capabilities'].items()],
            'modules':[{'name':name, 'available':item['available'], 'status':item['status'],
                        'version':item['schema_version']} for name,item in response['modules'].items()]}
        if step2:
            snapshot['wire_snapshot'] = {k:response[k] for k in ('catalog_version','catalog_digest','availability_digest','models')}
            snapshot['excluded_models'] = {m['id']:m['reason_code'] for m in response['models'] if m['id'] in state['task']['allowed_models'] and not m['available']}
        snapshot['snapshot_fingerprint'] = fingerprint({key:value for key,value in snapshot.items() if key != 'observed_at'})
        state = apply_patch(state, {'capabilities':snapshot})
        return _next(state, 'session')

    def check_locked(self, state, response):
        locked, task = response['locked_config'], state['task']
        expected = {'dataset_id':task['dataset_id'], 'selection_metric':task['selection_metric'],
            'allowed_models':state['capabilities']['eligible_models'], 'max_runs':1,
            'seed':task['seed'], 'evaluation_config':task['evaluation_config'], 'modules':[],
            'context_policy':{key:state['module_policy']['context_policy'][key]
                              for key in ('source_role','case_write')}}
        if state['versions']['state']=='agent-state-v3':
            expected['modules']=['train_evidence','legal_recipes']
            expected['context_policy'].update({k:state['module_policy']['context_policy'][k] for k in ('evidence','risks')})
        if any(locked[key] != value for key,value in expected.items()):
            raise ValueError('locked configuration mismatch')
        if locked['dataset_fingerprint_status'] != 'ready' or not locked['dataset_sha256']:
            raise ValueError('dataset metadata unavailable')
        extra = {'capabilities':{'frozen_snapshot':locked['capability_snapshot']}} if state['versions']['api'] == 'agent-session-v2' else {}
        if extra:
            from agent_poc.clients.contracts_v2 import FrozenSnapshot, validate_params
            frozen = FrozenSnapshot.model_validate(locked['capability_snapshot'])
            for name, overrides in task['model_configs'].items():
                if name in frozen.model_configs:
                    model = next(m for m in frozen.models if m.id == name)
                    resolved = {**model.fixed_execution_defaults, **validate_params(model,overrides)}
                    if resolved != frozen.model_configs[name]:
                        raise ValueError('operator parameters changed')
            self.deps.client.restore_frozen_session(response['session_id'], locked['capability_snapshot'], **({'preparation':locked['preparation']} if state['versions']['state']=='agent-state-v3' else {}))
        task_extra={}
        if state['versions']['state']=='agent-state-v3':
            prepared=locked['preparation']
            extra.update(evidence={'status':'ready','content':prepared['evidence']},
                recipes={'status':'ready','catalog':prepared['catalog'],'evaluation_plan':prepared['evaluation_plan']})
            task_extra=dict(split_fingerprint=prepared['evaluation_plan']['partition_digest'],split_fingerprint_status='ready')
        state = apply_patch(state, {**extra, 'task': {**task_extra, 'dataset_fingerprint':locked['dataset_sha256'],
            'dataset_fingerprint_status':'ready'}, 'identity':{'session_id':response['session_id']},
            'finalization':{'backend_session_state':response['state']},
            'recovery':{'last_confirmed_backend_state':{'session_state':response['state'],
                'selected_run_id':response.get('selected_run_id'), 'confirmed_at':self.deps.clock()}}})
        return self.remaining(state, response)

    def session(self, state):
        content = dict(state['recovery']['pending_operation']['content'])
        content.pop('evaluation')  # Client alone translates the fixed public request to HTTP.
        content['context_policy'] = {k:content['context_policy'][k] for k in (('source_role','case_write','evidence','risks') if state['versions']['state']=='agent-state-v3' else ('source_role','case_write'))}
        response = self.call(state, 'start_ml_session', content)
        state = self.check_locked(state, response)
        return _next(state, 'inspect_session')

    def bind_execution(self, state, response):
        keys = ('model_type','normalization','class_balance') + (('model_params',) if state['versions']['api'] == 'agent-session-v2' else ())
        action = {k:response['effective_action'][k] for k in keys}
        expected = state['execution']['submission_content']
        if state['versions']['state']=='agent-state-v3':
            recipe=next(r for r in state['recipes']['catalog']['recipes'] if r['recipe_id']==expected['recipe_id'])
            expected={**recipe['fixed_execution_config'],'model_params':state['capabilities']['frozen_snapshot']['model_configs'][recipe['model_id']]}
        if action != {k:expected[k] for k in action}:
            raise ValueError('backend action differs from submitted decision')
        if response['effective_config_status'] != 'ready' or response['dataset_fingerprint_status'] != 'ready':
            raise ValueError('Run metadata unavailable')
        if response['effective_config']['feature_selection_enabled']:
            raise ValueError('unexpected effective feature selection')
        return apply_patch(state, {'execution': {'run_id':response['run_id'],
            **({'resolved_execution':response['resolved_execution']} if 'resolved_execution' in response else {}),
            'run_status':response['state'], 'effective_action':action,
            'effective_config':response['effective_config'], 'effective_config_status':'ready',
            'dataset_fingerprint':response['dataset_sha256'], 'dataset_fingerprint_status':'ready'},
            'recovery':{'last_confirmed_backend_state':{'run_status':response['state']}}})

    def inspect_session(self, state):
        response = self.call(state, 'inspect_ml_session', {'session_id':state['identity']['session_id']})
        state = self.check_locked(state, response)
        if response['state'] == 'finalized':
            return self.lock_confirmed(state, response)
        experiments = response['experiments']
        if len(experiments) > 1:
            raise ValueError('multiple experiments violate protocol')
        if experiments:
            item = experiments[0]
            if item['binding_state'] == 'released':
                return _stop(state, 'agent_request_released', self.deps.clock())
            if item['binding_state'] in ('reserved','compensation_required'):
                return _next(state, 'reconcile', status='recovering')
            state = self.bind_execution(state, item)
            state = self.remaining(state, response)
            return _next(state, 'observe')
        return _next(state, 'submit' if state['execution']['submission_content'] else 'choose')

    def choose(self, state):
        context = selection_context(task=state['task'], models=state['capabilities']['eligible_models'],
            session_id=state['identity']['session_id'], client_request_id=state['identity']['experiment_request_id'],
            model_configs=state['capabilities']['frozen_snapshot']['model_configs'] if state['versions']['api'] == 'agent-session-v2' else None)
        if state['versions']['state']=='agent-state-v3':
            context=recipe_selection_context(task=state['task'],session_id=state['identity']['session_id'],
                preparation=self.preparation(state),context_policy=state['module_policy']['context_policy'])
        proposal = self.llm_call(state, 'submit', context)
        if proposal.tool_name != 'submit_ml_experiment':
            raise ValueError('wrong decision tool')
        request_type, decision_type = ExperimentRequest, DecisionState
        if state['versions']['api'] == 'agent-session-v2':
            from .state_v2 import ExperimentRequest as request_type, DecisionState as decision_type
        arguments=dict(proposal.arguments)
        if state['versions']['state']=='agent-state-v3':
            from .state import RecipeExperimentRequest as request_type
            if 'client_request_id' in arguments:
                raise ValueError('request ID is controlled by runner')
            arguments['client_request_id']=state['identity']['experiment_request_id']
        content = request_type.model_validate(arguments).model_dump(mode='json')
        if any(content[k] != value for k,value in context['bindings'].items()):
            raise ValueError('decision binding mismatch')
        recipe_profile=state['versions']['state']=='agent-state-v3'
        if not recipe_profile and content['model_type'] not in context['capabilities']['models']:
            raise ValueError('decision model not permitted')
        keys = ('model_type','normalization','class_balance') + (('model_params',) if state['versions']['api'] == 'agent-session-v2' else ())
        action = None if recipe_profile else {k:content[k] for k in keys}
        if recipe_profile and content['recipe_id'] not in {r['recipe_id'] for r in context['recipes']}:
            raise ValueError('recipe outside frozen catalog')
        decision = decision_type(status='ready', decision_id=_key('decision', state['identity']['task_id']),
            kind='recipe_selection' if recipe_profile else 'direct_action', action=action,
            **({'recipe_id':content['recipe_id']} if recipe_profile else {}), rationale=proposal.rationale, validation_status='ready',
            tool_name=proposal.tool_name, tool_call_id=proposal.tool_call_id,
            response_id=proposal.response_id).model_dump(mode='json')
        recipe_use={'recipes':{'uses':[dict(use_id=_key('recipe-use',state['identity']['task_id']),
            recipe_id=content['recipe_id'],decision_id=decision['decision_id'],experiment_id=state['identity']['current_experiment_id'])]}} if recipe_profile else {}
        state = apply_patch(state, {**recipe_use,'decision':decision,
            'execution':{'experiment_id':state['identity']['current_experiment_id'],
                'submission_request_id':content['client_request_id'], 'submission_content':content,
                'submission_fingerprint':fingerprint(content)},
            'guard':{'checks':[{'check_id':'decision-action', 'phase':'pre', 'kind':'action',
                'version':state['versions']['api'], 'status':'passed'}]}})
        return _next(state, 'submit')

    def submit(self, state):
        response = self.call(state, 'submit_ml_experiment', state['execution']['submission_content'])
        state = self.bind_execution(state, response)
        # Budget is confirmed by the following observation/session, not inferred from a POST.
        return _next(state, 'observe')

    def observe(self, state):
        response = self.call(state, 'observe_ml_experiment', {
            'session_id':state['identity']['session_id'], 'run_id':state['execution']['run_id']})
        if response['selection_metric'] != state['task']['selection_metric']:
            raise ValueError('selection metric changed')
        state = self.bind_execution(state, response)
        state = self.remaining(state, response)
        valid = response['validation']['status']
        if valid == 'failed':
            valid = 'invalid'
        progress = response['progress']
        projected_progress = {'stage':progress.get('stage') if isinstance(progress.get('stage'),str) else None,
            'percent':float(progress['percent']) if isinstance(progress.get('percent'),(float,int)) else None,
            'current':progress.get('epoch') if type(progress.get('epoch')) is int else None,
            'total':progress.get('epochs') if type(progress.get('epochs')) is int else None}
        error = response.get('error')
        state = apply_patch(state, {'feedback':{'status':'ready', 'run_status':response['state'],
            'validation_status':valid, 'validation_metrics':response['validation']['metrics'],
            'selection_score':response['validation_score'],
            'error':{'code':error['code'], 'retryable':error['retryable']} if error else None,
            'progress':projected_progress, 'allowed_actions':response['allowed_actions'],
            'integrity':'ready' if valid == 'ready' else 'pending' if valid == 'pending' else 'invalid',
            'retry_after_seconds':response.get('retry_after_seconds')},
            'execution':{'progress':projected_progress}})
        if response['state'] in ('queued','running'):
            return _next(state, 'observe', status='waiting',
                         wake=self.deps.clock()+response['retry_after_seconds'])
        if response['state'] in ('failed','cancelled'):
            return _stop(state, f"run_{response['state']}", self.deps.clock(), status=response['state'])
        eligible = valid == 'ready' and response['validation_score'] is not None and 'finalize_ml_session' in response['allowed_actions']
        checks = [{'check_id':'manifest-validation', 'phase':'post', 'kind':'manifest',
            'version':state['versions']['observation'], 'status':'passed' if valid == 'ready' else 'failed'},
            {'check_id':'selection-metric', 'phase':'post', 'kind':'selection_metric',
             'version':state['versions']['observation'], 'status':'passed' if eligible else 'failed'}]
        state = apply_patch(state, {'guard':{'checks':checks, 'candidate_eligible':eligible}})
        if not eligible:
            return _stop(state, 'no_valid_candidate', self.deps.clock())
        candidate = Candidate(candidate_id=f"candidate-{state['execution']['run_id']}",
            session_id=state['identity']['session_id'], run_id=state['execution']['run_id'],
            experiment_id=state['execution']['experiment_id'], status='valid',
            validation_metrics=response['validation']['metrics'], selection_score=response['validation_score']).model_dump(mode='json')
        state = apply_patch(state, {'candidates':{'status':'ready', 'items':[candidate]}})
        return _next(state, 'finalize_decision')

    def finalize_decision(self, state):
        context = finalization_context(task=state['task'], session_id=state['identity']['session_id'],
            run_id=state['execution']['run_id'], validation=state['feedback']['validation_metrics'],
            validation_score=state['feedback']['selection_score'], allowed_actions=state['feedback']['allowed_actions'],
            context_version=state['versions']['context_projection'])
        proposal = self.llm_call(state, 'finalize', context)
        if proposal.tool_name != 'finalize_ml_session' or proposal.arguments != context['bindings']:
            raise ValueError('invalid finalize decision')
        decision = DecisionState(status='ready', decision_id=_key('final-decision', state['identity']['task_id']),
            kind='finalize', selected_run_id=state['execution']['run_id'], rationale=proposal.rationale,
            validation_status='ready', tool_name=proposal.tool_name,
            tool_call_id=proposal.tool_call_id, response_id=proposal.response_id).model_dump(mode='json')
        state = apply_patch(state, {'decision':decision, 'finalization':{
            'selected_run_id':state['execution']['run_id'], 'rationale':proposal.rationale, 'status':'pending'},
            'candidates':{'recommendations':[{'recommendation_id':decision['decision_id'],
                'decision_id':decision['decision_id'], 'run_id':state['execution']['run_id'],
                'rationale':proposal.rationale}]}})
        return _next(state, 'finalize')

    def finalize(self, state):
        # Also runs after an OS kill before the POST response was checkpointed.
        response = self.call(state, 'inspect_ml_session', {'session_id':state['identity']['session_id']}, suffix='-inspect')
        state = self.check_locked(state, response)
        if response['state'] == 'finalized':
            return self.lock_confirmed(state, response)
        self.call(state, 'finalize_ml_session', state['recovery']['pending_operation']['content'])
        return _next(state, 'confirm')

    def confirm(self, state):
        response = self.call(state, 'inspect_ml_session', {'session_id':state['identity']['session_id']})
        state = self.check_locked(state, response)
        if response['state'] != 'finalized':
            return _next(state, 'finalize', status='recovering')
        return self.lock_confirmed(state, response)

    def lock_confirmed(self, state, response):
        if (not state['finalization']['selected_run_id'] or
                response['selected_run_id'] != state['finalization']['selected_run_id']):
            return _stop(state, 'backend_selection_conflict', self.deps.clock(), status='needs_attention', uncertain=True)
        locked_at = datetime.fromisoformat(response['finalized_at'].replace('Z','+00:00')).timestamp()
        return apply_patch(state, {'finalization':{'status':'confirmed', 'backend_session_state':'finalized',
            'locked_at':locked_at, 'termination_reason':'single_experiment_completed'},
            'lifecycle':{'status':'completed', 'stage':'confirmed', 'next_action':None,
                         'ended_at':self.deps.clock(), 'reason_code':'single_experiment_completed'},
            'recovery':{'pending_operation':None, 'next_wake_at':None, 'needs_human_review':False}})

    def reconcile(self, state):
        # Never exposed to the LLM; the server owns staleness and resolution.
        response = self.call(state, 'reconcile_ml_session', {'session_id':state['identity']['session_id']})
        if len(response['items']) > 1:
            raise ValueError('multiple reservations violate protocol')
        item = response['items'][0] if response['items'] else None
        if item is None:
            return _next(state, 'inspect_session', status='recovering')
        outcome = {'reserved':'pending'}.get(item['state'], item['state'])
        state = apply_patch(state, {'recovery':{'reconciliation':{'status':'ready',
            'outcome':outcome, 'resolution_code':item['resolution_code'], 'checked_at':self.deps.clock()}}})
        if item['state'] == 'released':
            return _stop(state, 'agent_request_released', self.deps.clock())
        if item['requires_manual_review']:
            return _stop(state, item['resolution_code'], self.deps.clock(), status='needs_attention', uncertain=True)
        if item['state'] == 'bound':
            return _next(state, 'inspect_session', status='recovering')
        # One check per server stale interval; finite API limit and task deadline survive restarts.
        return _next(state, 'reconcile', status='waiting', reason=item['resolution_code'], wake=self.deps.clock()+300)


def build_graph(dependencies: Dependencies, checkpointer):
    nodes = Nodes(dependencies)
    graph = StateGraph(GraphState)
    graph.add_node('prepare', nodes.prepare)
    graph.add_edge(START, 'prepare')
    for action in TOOLS:
        graph.add_node('do_' + action, lambda state, action=action: nodes.execute(action, state))
        graph.add_edge('do_' + action, END)

    def route(state):
        action = state['lifecycle']['next_action']
        wake = state['recovery']['next_wake_at']
        return END if action is None or (wake is not None and dependencies.clock() < wake) else action

    graph.add_conditional_edges('prepare', route, {**{action:'do_' + action for action in TOOLS}, END:END})
    return graph.compile(checkpointer=checkpointer, name='agent-single-experiment-v1')
