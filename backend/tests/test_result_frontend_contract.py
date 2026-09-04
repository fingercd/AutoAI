from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[2]


def _run_node(source: str) -> None:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(source)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_sample_id_view_model_collapses_at_ten_and_reports_exact_remainder() -> None:
    _run_node(
        """
        import assert from 'node:assert/strict';
        const ui = await import('./static/js/ui-utils.js');
        assert.equal(ui.DEFAULT_SAMPLE_ID_LIMIT, 10);

        const makeGroups = (count) => Array.from({ length: count }, (_, index) => ({
          sample_id: String(index + 1),
          label: index % 2 ? 'B' : 'A',
          count: 3,
        }));

        const ten = ui.buildSampleIdViewModel(makeGroups(10), false);
        assert.equal(ten.visible.length, 10);
        assert.equal(ten.remaining, 0);

        const eleven = ui.buildSampleIdViewModel(makeGroups(11), false);
        assert.equal(eleven.visible.length, 10);
        assert.equal(eleven.remaining, 1);
        assert.equal(eleven.buttonText, '展开其余 1 个');

        const fortySix = ui.buildSampleIdViewModel(makeGroups(46), false);
        assert.equal(fortySix.visible.length, 10);
        assert.equal(fortySix.remaining, 36);
        assert.equal(fortySix.buttonText, '展开其余 36 个');

        const expanded = ui.buildSampleIdViewModel(makeGroups(46), true);
        assert.equal(expanded.visible.length, 46);
        assert.equal(expanded.buttonText, '收起');

        const naturallySorted = ui.buildSampleIdViewModel([
          { sample_id: '10' }, { sample_id: '2' }, { sample_id: '1' },
        ], true);
        assert.deepEqual(naturallySorted.visible.map((item) => item.sample_id), ['1', '2', '10']);
        """
    )


def test_curve_index_parser_accepts_common_forms_and_caps_large_selections() -> None:
    _run_node(
        """
        import assert from 'node:assert/strict';
        const ui = await import('./static/js/ui-utils.js');

        assert.equal(ui.MAX_CURVE_SELECTION, 5000);
        assert.deepEqual([...ui.parseCurveIndexExpression('1,3,5')], ['1', '3', '5']);
        assert.deepEqual([...ui.parseCurveIndexExpression('1-5')], ['1', '2', '3', '4', '5']);
        assert.throws(
          () => ui.parseCurveIndexExpression('1-999999999'),
          /最多选择5000条曲线/,
        );
        const tooManyUnique = Array.from({ length: 5001 }, (_, index) => index + 1).join(',');
        assert.throws(
          () => ui.parseCurveIndexExpression(tooManyUnique),
          /最多选择5000条曲线/,
        );
        """
    )


def test_hplc_row_range_validator_is_strict_and_reports_actual_point_count() -> None:
    _run_node(
        """
        import assert from 'node:assert/strict';
        const ui = await import('./static/js/ui-utils.js');

        assert.equal(ui.HPLC_POINT_COUNT, undefined);
        assert.deepEqual(ui.validateHplcRowRange('100', '4000', 8000), {
          startRow: 100,
          endRow: 4000,
          pointCount: 3901,
        });
        assert.deepEqual(ui.validateHplcRowRange('1', '', 8000), {
          startRow: 1,
          endRow: 8000,
          pointCount: 8000,
        });
        assert.throws(() => ui.validateHplcRowRange('0', '10', 8000), /1–8000/);
        assert.throws(() => ui.validateHplcRowRange('1', '8001', 8000), /不能超过 8000/);
        assert.throws(() => ui.validateHplcRowRange('1', '9000', 8000), /不能超过 8000/);
        assert.throws(() => ui.validateHplcRowRange('10.5', '20', 8000), /必须是整数/);
        assert.throws(() => ui.validateHplcRowRange('4000', '100', 8000), /不能大于终止行/);
        assert.throws(() => ui.validateHplcRowRange('1', ''), /完成 HPLC 文件点数检测/);
        assert.deepEqual(ui.validateHplcRowRange('1', '5', 5), {
          startRow: 1,
          endRow: 5,
          pointCount: 5,
        });
        """
    )


def test_hash_result_route_and_auto_redirect_are_pure_and_run_specific() -> None:
    _run_node(
        """
        import assert from 'node:assert/strict';
        const results = await import('./static/js/run-results.js');

        assert.equal(results.AUTO_REDIRECT_DELAY_MS, 3000);
        assert.deepEqual(results.parseHash('#/results?run_id=run%2F%E4%B8%AD%E6%96%87'), {
          view: 'results',
          runId: 'run/中文',
        });
        assert.deepEqual(results.parseHash('#/comparison?batch_id=batch%2F%E4%B8%AD%E6%96%87'), {
          view: 'comparison',
          runId: null,
          batchId: 'batch/中文',
        });
        assert.equal(results.buildViewHash('runs'), '#/runs');
        assert.equal(results.buildResultHash('run/中文'), '#/results?run_id=run%2F%E4%B8%AD%E6%96%87');
        assert.equal(results.buildComparisonHash('batch/中文'), '#/comparison?batch_id=batch%2F%E4%B8%AD%E6%96%87');
        assert.equal(results.shouldAutoRedirect({ runId: 'run-1', state: 'succeeded', isNewRun: true }), true);
        assert.equal(results.shouldAutoRedirect({ runId: 'run-1', state: 'running', isNewRun: true }), false);
        assert.equal(results.shouldAutoRedirect({ runId: 'run-1', state: 'succeeded', isNewRun: false }), false);
        assert.equal(results.shouldAutoRedirect({ runId: '', state: 'succeeded', isNewRun: true }), false);
        assert.equal(results.backendSupportsResultContract({
          contracts: { run_result: 'run-result-v1' },
        }), true);
        assert.equal(results.backendSupportsResultContract({ contracts: {} }), false);
        assert.equal(results.shouldUseLegacyResultFallback(
          { status: 404 },
          { contracts: { run_result: 'legacy' } },
        ), true);
        assert.equal(results.shouldUseLegacyResultFallback(
          { status: 404 },
          { contracts: { run_result: 'run-result-v1' } },
        ), false);
        assert.equal(results.shouldUseLegacyResultFallback({ status: 404 }, null), false);
        assert.doesNotMatch(results.trainingTimeText({
          created_at: '2026-07-21T01:00:00Z',
          started_at: '2026-07-21T02:00:00Z',
        }), /任务创建/);
        assert.match(results.trainingTimeText({
          created_at: '2026-07-21T01:00:00Z',
          started_at: null,
        }), /任务创建/);
        assert.equal(results.trainingTimeText({}), '—');
        """
    )


def test_result_split_analysis_order_and_vertical_scale_are_deterministic() -> None:
    _run_node(
        """
        import assert from 'node:assert/strict';
        const results = await import('./static/js/run-results.js');

        const entries = results.analysisSplitEntries({
          analysis: {
            splits: {
              test: { aggregation: 'pooled_oof' },
              train: { aggregation: 'pooled_cross_fold' },
              valid: { aggregation: 'pooled_cross_fold' },
            },
          },
        });
        assert.deepEqual(entries.map((entry) => entry.name), ['train', 'valid', 'test']);
        assert.deepEqual(results.analysisSplitEntries({
          analysis: { confusion_matrix: [[1]] },
        }).map((entry) => entry.name), ['test']);
        assert.deepEqual(results.distributionScaleTicks(0), [0]);
        assert.deepEqual(results.distributionScaleTicks(6), [6, 4, 2, 0]);
        """
    )


def test_server_token_is_session_only_and_sent_only_to_same_origin_api() -> None:
    _run_node(
        """
        import assert from 'node:assert/strict';
        const values = new Map();
        const sessionStorage = {
          getItem: (key) => values.get(key) ?? null,
          setItem: (key, value) => values.set(key, String(value)),
          removeItem: (key) => values.delete(key),
        };
        globalThis.window = {
          sessionStorage,
          location: { href: 'https://autoai.example.edu/', origin: 'https://autoai.example.edu' },
          dispatchEvent: () => {},
        };
        Object.defineProperty(window, 'localStorage', {
          get() { throw new Error('localStorage must not be accessed'); },
        });

        const observed = [];
        globalThis.fetch = async (url, options) => {
          observed.push({ url: String(url), authorization: options.headers.get('Authorization') });
          return new Response(JSON.stringify({ ok: true }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        };
        const api = await import('./static/js/api-client.js');
        assert.equal(api.SERVER_TOKEN_STORAGE_KEY, 'specautoai.serverToken');
        api.setServerToken('server-secret');
        assert.equal(api.getServerToken(), 'server-secret');

        await api.request('/api/training/runs', { authRetry: false });
        await api.request('https://other.example/api/training/runs', { authRetry: false });
        assert.equal(observed[0].authorization, 'Bearer server-secret');
        assert.equal(observed[1].authorization, null);

        api.clearServerToken();
        assert.equal(api.getServerToken(), '');
        """
    )


def test_result_ui_has_a_dedicated_view_and_does_not_advertise_unavailable_roc_auc() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-view="results"' in html
    assert 'id="view-results"' in html
    assert html.index('data-view="modeling"') < html.index('data-view="results"') < html.index('data-view="runs"')
    for element_id in (
        "resultPageState",
        "resultOverview",
        "resultMetrics",
        "resultSplitMetrics",
        "resultAnalysis",
        "resultArtifacts",
    ):
        assert f'id="{element_id}"' in html

    assert "rocCanvas" not in html
    assert "precisionRecallCanvas" not in html
    assert "aucMetricCard" not in html
    results = (ROOT / "static" / "js" / "run-results.js").read_text(encoding="utf-8")
    assert "downloadConfusionMatrixPng" in results
    assert "renderModelFeatureVisualization" not in results
    assert "TEMPORARILY_HIDDEN" in results


def test_confusion_png_filename_and_layout_are_safe_and_split_specific() -> None:
    _run_node(
        """
        import assert from 'node:assert/strict';
        const png = await import('./static/js/confusion-matrix-download.js');
        assert.equal(png.confusionMatrixPngFilename('run/中文:42', 'valid'), 'run_run_42__confusion_matrix_valid.png');
        assert.equal(png.confusionMatrixPngFilename('abc', 'test'), 'run_abc__confusion_matrix_test.png');
        const layout = png.confusionMatrixCanvasLayout([[4, 1], [2, 3]], ['甲类', '乙类']);
        assert.equal(layout.size, 2);
        assert.deepEqual(layout.matrix, [[4, 1], [2, 3]]);
        assert.ok(layout.width > 0 && layout.height > 0);
        """
    )


def test_each_primary_page_has_short_contextual_help_and_global_help_stays_small() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    for title in (
        'AI 建模操作说明',
        '拉曼预处理操作说明',
        '色谱预处理操作说明',
        '建模结果查看说明',
        '训练记录操作说明',
    ):
        assert f'<summary>{title}</summary>' in html
    assert html.count('<details class="page-help">') >= 5
    assert '<h2>全局使用说明</h2>' in html
    assert '各功能页顶部都提供与当前操作直接相关的简短说明' in html


def test_known_untrusted_fields_no_longer_flow_into_inline_html_or_handlers() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    ui = (ROOT / "static" / "js" / "ui-utils.js").read_text(encoding="utf-8")
    results = (ROOT / "static" / "js" / "run-results.js").read_text(encoding="utf-8")

    assert "node.textContent = String(value)" in ui
    assert "sampleIdRow(model.visible[index])" in ui
    assert "onclick=" not in html
    assert '${run.error || "请查看后台日志"}' not in html
    assert 'curveSelect").innerHTML = curves.map' not in html
    assert "<th>${label}</th>" not in html
    assert "innerHTML" not in results
