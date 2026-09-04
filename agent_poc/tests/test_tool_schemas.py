from agent_poc.tools import TOOL_SCHEMAS


def test_tool_schemas_are_closed_and_have_no_path_or_test_fields():
    assert len(TOOL_SCHEMAS) == 6
    for schema in TOOL_SCHEMAS.values():
        assert schema['additionalProperties'] is False
        fields = {name.lower() for name in schema['properties']}
        assert not any('path' in name or 'test' in name for name in fields)


def test_model_inputs_are_enumerated():
    experiment = TOOL_SCHEMAS['submit_ml_experiment']
    assert experiment['properties']['model_type']['enum'] == [
        'logistic_regression', 'svm', 'random_forest'
    ]
