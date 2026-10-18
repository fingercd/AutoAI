"""学生界面只隐藏控制入口，不移除结果图表和后台能力。"""
from html.parser import HTMLParser
from pathlib import Path


class VisibilityParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.hidden_ids = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = any(item[1] for item in self.stack) or 'hidden' in attrs
        if 'id' in attrs:
            self.hidden_ids[attrs['id']] = hidden
        if tag not in {'input', 'meta', 'link', 'br', 'hr', 'img'}:
            self.stack.append((tag, hidden))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def test_student_controls_are_hidden_without_hiding_modeling_and_hplc_actions():
    parser = VisibilityParser()
    parser.feed(Path('static/index.html').read_text(encoding='utf-8'))
    for id_ in ['trainingProfile', 'splitTrain', 'splitValid', 'splitTest',
                'cvEnabled', 'advancedToggle', 'normalization', 'seed', 'epochs',
                'ramanBaselineMethodMode', 'ramanCustomMethod']:
        assert parser.hidden_ids[id_] is True, id_
    for id_ in ['featureScheme', 'startTrain', 'modelTypeChoices', 'datasetFileTrigger',
                'testDatasetFileTrigger', 'chromRangeMode', 'chromHplcInterp',
                'ramanRangeMode', 'chromPreprocessButton']:
        assert parser.hidden_ids[id_] is False, id_


def test_history_removes_copy_id_and_keeps_state_specific_actions():
    html = Path('static/index.html').read_text(encoding='utf-8')
    history = html.split('async function loadRuns(', 1)[1].split('function clearDeletedRunPanel', 1)[0]
    assert '复制 ID' not in history
    assert 'if (active) actionWrap.append(stopButton)' in history
    assert 'if (terminal) actionWrap.append(deleteButton)' in history
    assert '查看所属对比' in history


def test_result_technical_details_and_audits_remain_collapsed():
    results = Path('static/js/run-results.js').read_text(encoding='utf-8')
    assert "element('details', {},\n    element('summary', { text: '任务技术详情' })" in results
    assert "overviewItem('Run ID', runId, copyButton(runId))" in results
    assert "text: '按任务 ID 查找'" in results
    assert "element('details', { className: 'result-card wide-card audit-details' }" in results
