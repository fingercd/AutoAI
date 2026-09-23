"""Finite, versioned controls over existing training algorithms.

Search grids and network profiles remain owned by the trainer. This module
only describes their resolution policy and the inputs operators may freeze.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Any

from .classification_policy import DEEP_TRAINING_DEFAULTS
from .model_catalog import MODELS_BY_ID, MODEL_DECLARATIONS, ARCHITECTURE_VERSION, model_availability


@dataclass(frozen=True)
class TraditionalTrainingDefaults:
    random_forest_n_estimators: int = 200
    random_forest_search_iterations: int = 10
    xgboost_gamma: float = 0.0


TRADITIONAL_TRAINING_DEFAULTS = TraditionalTrainingDefaults()
CONFIG_POLICY_VERSION = 'agent-model-config-v1'
CATALOG_VERSION = 'agent-model-catalog-v1'


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _parameter(name, kind, default, *, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, choices=None):
    return dict(**({'exclusive_maximum': exclusive_maximum} if exclusive_maximum is not None else {}), name=name, value_type=kind, nullable=False, default_status='fixed', default=default,
                minimum=minimum, maximum=maximum, exclusive_minimum=exclusive_minimum, choices=choices,
                role='operator_fixed', applies_to=[], condition_refs=[])


def model_policy(model_id: str) -> dict[str, Any]:
    model = MODELS_BY_ID[model_id]
    parameters = []
    rules = []
    if model.implemented and model.execution_family == 'deep_learning':
        for name, default in asdict(DEEP_TRAINING_DEFAULTS).items():
            if name == 'seed':
                continue
            if type(default) is int:
                parameters.append(_parameter(name, 'integer', default, minimum=1))
            else:
                parameters.append(_parameter(name, 'number', default,
                    minimum=0 if name == 'weight_decay' else None,
                    exclusive_minimum=None if name == 'weight_decay' else 0,
                    exclusive_maximum=1 if name == 'scheduler_factor' else None))
        rules.append('train-only-profile-v2')
    defaults = TRADITIONAL_TRAINING_DEFAULTS
    if model_id == 'random_forest':
        parameters += [_parameter('random_forest_n_estimators', 'integer', defaults.random_forest_n_estimators, minimum=50, maximum=1000),
                       _parameter('random_forest_search_iterations', 'integer', defaults.random_forest_search_iterations, minimum=1, maximum=18)]
        rules.append('rf-oob-balanced-accuracy-accuracy-stable-order-v1')
    elif model_id == 'xgboost':
        parameters.append(_parameter('xgboost_gamma', 'number', defaults.xgboost_gamma, minimum=0))
        rules.append('xgboost-twelve-candidate-validation-search-v2')
    elif model.execution_family == 'traditional_ml':
        rules.append(model_id + '-train-bounded-validation-search-v2')
    for parameter in parameters:
        parameter['applies_to'] = [model_id]
    policy = dict(config_policy_version='agent-model-config-v2' if model.implemented and model.execution_family == 'deep_learning' else CONFIG_POLICY_VERSION,
        fixed_execution_defaults={p['name']: p['default'] for p in parameters},
        parameters=parameters, compatibility_rules=rules)
    policy['config_policy_digest'] = semantic_digest(policy)
    return policy


def resolve_model_params(model_id: str, overrides: dict[str, Any] | None = None,
                         *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = model_policy(model_id) if policy is None else policy
    if overrides is None:
        overrides = {}
    if type(overrides) is not dict:
        raise ValueError('model_params must be an object')
    specs = {p['name']:p for p in policy['parameters'] if p['role'] == 'operator_fixed'}
    if set(overrides) - set(specs):
        raise ValueError('unknown or inapplicable model parameter')
    resolved = dict(policy['fixed_execution_defaults'])
    for name, value in overrides.items():
        spec = specs[name]
        kind = spec['value_type']
        if value is None or (kind == 'integer' and type(value) is not int) or (
            kind == 'number' and type(value) not in (float, int)) or (
            kind == 'string' and type(value) is not str):
            raise ValueError('invalid parameter type: ' + name)
        if kind in ('integer', 'number'):
            if not math.isfinite(value):
                raise ValueError('parameter must be finite: ' + name)
            for key, invalid in [('minimum', lambda x: value < x), ('maximum', lambda x: value > x),
                                  ('exclusive_minimum', lambda x: value <= x), ('exclusive_maximum', lambda x: value >= x)]:
                if spec.get(key) is not None and invalid(spec[key]):
                    raise ValueError('parameter out of range: ' + name)
        if spec['choices'] is not None and value not in spec['choices']:
            raise ValueError('invalid parameter choice: ' + name)
        resolved[name] = float(value) if kind == 'number' else value
    return resolved


def model_capability_snapshot() -> dict[str, Any]:
    models = []
    semantic = []
    availability = []
    for m in MODEL_DECLARATIONS:
        item = dict(id=m.id, display_name=m.display_name, family=m.family,
                    execution_family=m.execution_family, architecture_version=ARCHITECTURE_VERSION,
                    implemented=m.implemented, **model_policy(m.id))
        semantic.append(dict(item))
        available, reason = model_availability(m.id)
        availability.append(dict(id=m.id, available=available, reason_code=reason))
        item.update(available=available, reason_code=reason, availability_basis='declaration-module-dependency-probe')
        models.append(item)
    return dict(catalog_version=CATALOG_VERSION, catalog_digest=semantic_digest(semantic),
                availability_digest=semantic_digest(availability), models=models)


def compatible_frozen_policy(model_id, frozen, params, *, current=None):
    """Only the known v1 scheduler boundary correction is execution compatible."""
    from copy import deepcopy
    current = model_policy(model_id) if current is None else current
    if frozen['config_policy_digest'] != current['config_policy_digest']:
        if current['config_policy_version'] != 'agent-model-config-v2':
            return False
        legacy = deepcopy(current)
        legacy.pop('config_policy_digest')
        legacy['config_policy_version'] = 'agent-model-config-v1'
        for parameter in legacy['parameters']:
            if parameter['name'] == 'scheduler_factor':
                parameter.pop('exclusive_maximum')
                parameter['maximum'] = 1
        if frozen['config_policy_digest'] != semantic_digest(legacy):
            return False
    try:
        return resolve_model_params(model_id, params) == params
    except ValueError:
        return False


def search_strategy_binding(model_id: str) -> dict[str, str]:
    """Bind reachable execution code specialized to one canonical model.

    Model dispatch branches are resolved before following local symbol imports.
    Adding an unrelated model therefore cannot invalidate an existing recipe.
    Unknown data-dependent branches stay in the digest conservatively.
    """
    import ast
    import copy
    from pathlib import Path
    from . import model_catalog

    model = MODELS_BY_ID[model_id]
    root = Path(__file__).parent
    model_sets = {name: value for name, value in vars(model_catalog).items()
                  if name.endswith('MODEL_TYPES') and isinstance(value, set)}

    class Specialize(ast.NodeTransformer):
        def __init__(self, symbols):
            self.model_sets = dict(model_sets)
            for name, node in symbols.items():
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    try: value = ast.literal_eval(node.value)
                    except (ValueError, TypeError): continue
                    if isinstance(value, (set, tuple, list)) and all(isinstance(x, str) for x in value):
                        self.model_sets[name] = value

        @staticmethod
        def model_value(node):
            return ((isinstance(node, ast.Name) and node.id in ('model_type','model_key'))
                    or (isinstance(node, ast.Attribute) and node.attr == 'model_type'))

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == 'canonical_model_type'
                    and len(node.args) == 1 and self.model_value(node.args[0])):
                return ast.copy_location(ast.Constant(model_id), node)
            return self.generic_visit(node)

        def truth(self, test):
            if not isinstance(test, ast.Compare) or len(test.ops) != 1:
                return None
            left = test.left
            if not (self.model_value(left) or isinstance(left, ast.Constant) and left.value == model_id):
                return None
            right = test.comparators[0]
            try:
                value = self.model_sets[right.id] if isinstance(right, ast.Name) and right.id in self.model_sets else ast.literal_eval(right)
            except (ValueError, TypeError):
                return None
            op = test.ops[0]
            if isinstance(op, ast.Eq): return model_id == value
            if isinstance(op, ast.NotEq): return model_id != value
            if isinstance(op, ast.In): return model_id in value
            if isinstance(op, ast.NotIn): return model_id not in value
            return None

        def visit_If(self, node):
            node = self.generic_visit(node)
            known = self.truth(node.test)
            return node if known is None else (node.body if known else node.orelse)

        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            known = self.truth(node.test)
            return node if known is None else (node.body if known else node.orelse)

        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
            return node

        visit_ClassDef = visit_FunctionDef

    parsed = {}; bound = {}; visiting = set()

    def source(path):
        if path not in parsed:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            symbols = {}; imports = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    symbols[node.name] = node
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name): symbols[target.id] = node
                elif isinstance(node, ast.ImportFrom) and node.level:
                    parent = path.parent
                    for _ in range(node.level - 1): parent = parent.parent
                    module = parent.joinpath(*(node.module or '').split('.')).with_suffix('.py')
                    if not module.is_file(): module = module.with_suffix('') / '__init__.py'
                    if module.is_file() and module.is_relative_to(root):
                        for alias in node.names: imports[alias.asname or alias.name] = (module, alias.name)
            parsed[path] = symbols, imports
        return parsed[path]

    def follow(path, name):
        key = (path, name)
        if key in visiting: return
        visiting.add(key)
        symbols, imports = source(path)
        if name not in symbols:
            if name in imports: follow(*imports[name])
            return
        node = Specialize(symbols).visit(copy.deepcopy(symbols[name]))
        bound[path.relative_to(root).as_posix() + ':' + name] = ast.dump(node, include_attributes=False)
        for dependency in ast.walk(node):
            if isinstance(dependency, ast.Name) and dependency.id != name:
                follow(path, dependency.id)
            elif isinstance(dependency, ast.ImportFrom) and dependency.level:
                parent = path.parent
                for _ in range(dependency.level - 1): parent = parent.parent
                module = parent.joinpath(*(dependency.module or '').split('.')).with_suffix('.py')
                if not module.is_file(): module = module.with_suffix('') / '__init__.py'
                if module.is_file() and module.is_relative_to(root):
                    for alias in dependency.names: follow(module, alias.name)

    names = ({'_fit_deep_fold'} if model.execution_family == 'deep_learning' else
        {'_traditional_candidate_configs', '_select_random_forest_config' if model_id == 'random_forest'
         else '_select_traditional_config', '_fit_final_traditional_model'})
    names.update(('_fit_x_normalizer', '_transform_x_with_normalizer'))
    for name in sorted(names):
        if name not in source(root / 'training.py')[0]: raise ValueError('search implementation unavailable')
        follow(root / 'training.py', name)
    if not any(key.startswith('models/') for key in bound):
        raise ValueError('model construction binding unavailable')
    rules = model_policy(model_id)['compatibility_rules']
    return dict(ref=rules[-1], version='training-execution-policy-v1',
                digest=semantic_digest(dict(revision='training-execution-policy-v1', implementation=bound)))


# Exact source bindings from the last accepted default-processing revision.
# Only the zscore/none path was equivalent across the normalizer repair.
_DEFAULT_PROCESSING_BINDING_COMPAT = {
    'cnn1d': ('d835450df23a3c40afa441773e02fccae734c5a48ec0ee5ab0d4c0b662683199', 'e09ec1f1e2ef530c378775afc75610914405467df9efa4e00ee77284625ae296'),
    'cnn1d_se': ('70f638a97937fac8adf6f7346d6d8c442e0f070917c79221d8bbaaa33eda3804', '66159230193dbc27c7f7280a2a7dd954449c66ba16f29c7fcc90b5cd735520d5'),
    'cnn_transformer1d': ('bc74fe86e9a916d2328d52dd4b16a1f87a165dcfc5799074b6e582877459fb6a', '3004bd3f4e32f3c76a32369e02640329c2432fccb472223aa0b270c2af46c42f'),
    'inception1d': ('08aa449ed5a2be1e47e953aa4546ccf13f94b26d746366784771df68122ae4c2', 'ad6e0f58858fd45b03b959f492eaf3a6129695bacc4dd47a4930241098e3c760'),
    'logistic_regression': ('98f25de6c56a875a5f76665bad4b4cfd271a6dd9373a800ea775f946e3fd93d2', 'd1ea85d6041056dfd9e8a936737141f41622e347ea192e1687821aa753cc6ed3'),
    'pca_lda': ('ba762ea4737eb065e0caa6cb54e0de850b233b271f6eddd91258d5f384a8b67a', '4992d8c518c381b48eb6ddb964264192a80e06316cead1a5ec3e23f278034b8b'),
    'pca_mlp': ('b49fd4f273ee7af345d9b35a83aa203042f8cb323605afcaf6b9ec2b24a7bf37', '7209c3b8eaa7b7df70b4b32774773d49bcb963a35cd68672f943c5db263e64e2'),
    'pls_da': ('ce30dff0bb2e96bc1902d2d0f56d070c07d2bc800bcec345156b299b2f1ed6e9', '05bb26ba8cf8be990bfa4a9fbc587aa4208781f5899ce8c5b69f85a801107b69'),
    'random_forest': ('299ca52e525de3bf5dc841208df9a15d33e0e2ed4c8d895b2d0bc0280ed12594', '401861090e63506c17d8cfefcc14d667ad61f112873797b199f1ec954acfd753'),
    'resnet1d': ('db67cb26bb0b8a90e35073d4fbca6b257a714b9de1834610e10e946ea0530554', '514a8342d28ea3ceb82a50c5f5c0ce5d85a009ae0353abad14f8c50701ad7987'),
    'svm': ('d0dd2b72459383c15e9e9750b6d80d155eb2b9419db6a4d47d5b727eb3388a31', 'e522d91e4f9dfc3c1ec1ebe6270eed051740de701be0ab8ac6594f86cb8a013b'),
    'tcn1d': ('eeb68e65877402286a4ab479edb34b25f947dc04e8cfbc516fe031cf3a492db6', 'f0c71ae4c5e8a4aa7fd5fdb18632fa7dc158f63a8b6161a975f8d524d99632e0'),
    'xgboost': ('34db4c1acf3bbb9511a1f2acfc103d79e6738c9350653ed5415dcb0fa657bfea', 'f9e28eb729ee27249e7e3b2e79707fd9c06f3f45e90f0521fba12f81cac73600'),
}


def compatible_search_strategy_binding(model_id: str, frozen_digest: str,
                                       normalization: str, class_balance: str) -> bool:
    current = search_strategy_binding(model_id)['digest']
    if frozen_digest == current:
        return True
    return (normalization, class_balance) == ('zscore', 'none') and (
        frozen_digest, current
    ) == _DEFAULT_PROCESSING_BINDING_COMPAT.get(model_id)
