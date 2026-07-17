/** v2 前端纯函数测试：node static/v2/tests/run-tests.mjs */
import assert from 'node:assert/strict';

import {
  RUN_STATE_META,
  LEGACY_STATUS_BY_STATE,
  RESULT_STATE_META,
  EVALUATION_STRATEGIES,
  stateMeta,
  resultStateMeta,
  isTerminalState,
  isActiveState,
  resolveSplitScalars,
  validateCvAggregation,
  findServerPaths,
  looksLikeServerPath,
  formatMetric,
  formatBytes,
  formatDuration,
} from '../lib/format.js';
import { createPoller } from '../lib/poller.js';
import { decimateSeries, niceTicks } from '../lib/charts.js';
import { escapeHtml } from '../lib/dom.js';
import {
  visibleArtifacts,
  artifactActionState,
  isSafeDownloadUrl,
  isForbiddenArtifactName,
} from '../components/artifacts.js';
import { normalizeMatrix, cellIntensity, matrixTotals } from '../components/confusion-matrix.js';
import { resolveSampleSeries } from '../components/explainability-panel.js';

const tests = [];
const test = (name, fn) => tests.push([name, fn]);
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

function fakeScheduler() {
  let nextId = 0;
  const pending = new Map();
  return {
    set(fn) {
      nextId += 1;
      pending.set(nextId, fn);
      return nextId;
    },
    clear(id) {
      pending.delete(id);
    },
    runNext() {
      const entry = pending.entries().next().value;
      if (!entry) return false;
      pending.delete(entry[0]);
      entry[1]();
      return true;
    },
    get size() {
      return pending.size;
    },
  };
}

// ---------- 状态映射 ----------

test('run.state 五态映射与旧兼容状态对照', () => {
  assert.deepEqual(Object.keys(RUN_STATE_META).sort(), ['cancelled', 'failed', 'queued', 'running', 'succeeded']);
  assert.equal(LEGACY_STATUS_BY_STATE.queued, 'pending');
  assert.equal(LEGACY_STATUS_BY_STATE.running, 'running');
  assert.equal(LEGACY_STATUS_BY_STATE.succeeded, 'success');
  assert.equal(LEGACY_STATUS_BY_STATE.failed, 'failed');
  assert.equal(LEGACY_STATUS_BY_STATE.cancelled, 'paused');
  assert.equal(stateMeta('queued').label, '排队中');
  assert.equal(stateMeta('未知状态').label, '未知状态');
  assert.ok(isTerminalState('succeeded') && isTerminalState('failed') && isTerminalState('cancelled'));
  assert.ok(!isTerminalState('queued') && !isTerminalState('running'));
  assert.ok(isActiveState('queued') && isActiveState('running') && !isActiveState('succeeded'));
});

test('result_state 八态映射齐备', () => {
  assert.deepEqual(
    Object.keys(RESULT_STATE_META).sort(),
    ['cancelled', 'corrupt_manifest', 'failed', 'missing_manifest', 'partial', 'pending', 'ready', 'running'],
  );
  assert.equal(resultStateMeta('ready').tone, 'success');
  assert.equal(resultStateMeta('corrupt_manifest').tone, 'danger');
  assert.ok(resultStateMeta('partial').description.length > 0);
});

test('三种评估口径常量齐备', () => {
  assert.deepEqual(
    Object.keys(EVALUATION_STRATEGIES).sort(),
    ['external_test_holdout', 'leave_one_sample_id_cv', 'stratified_holdout'],
  );
  for (const meta of Object.values(EVALUATION_STRATEGIES)) {
    assert.ok(meta.label && meta.description);
  }
});

// ---------- CV 口径守卫 ----------

const cvResult = {
  schema_version: 'run-result-v1',
  evaluation: { strategy: 'leave_one_sample_id_cv', fold_count: 3, primary_split: 'test', primary_aggregation: 'pooled_oof' },
  metrics: {
    primary: { accuracy: 0.8, macro_f1: 0.81 },
    pooled_oof: { accuracy: 0.8, macro_f1: 0.81 },
    splits: {
      train: { aggregation: 'fold_mean', values: { accuracy: 0.9 }, fold_std: { accuracy: 0.02 } },
      valid: { aggregation: 'fold_mean', values: { accuracy: 0.85 }, fold_std: { accuracy: 0.03 } },
      test: { aggregation: 'pooled_oof', values: { accuracy: 0.8, macro_f1: 0.81 }, fold_std: null },
    },
    fold_mean: { train: { accuracy: 0.9 }, valid: { accuracy: 0.85 }, test: { accuracy: 0.79, macro_f1: 0.5 } },
    fold_std: {},
  },
  analysis: {
    splits: {
      train: { aggregation: 'pooled_cross_fold' },
      valid: { aggregation: 'pooled_cross_fold' },
      test: { aggregation: 'pooled_oof' },
    },
  },
};

test('CV 的 Test 标量只取 pooled OOF，不取 fold mean', () => {
  const resolved = resolveSplitScalars(cvResult, 'test');
  assert.equal(resolved.aggregation, 'pooled_oof');
  assert.equal(resolved.values.macro_f1, 0.81);
  assert.equal(resolved.values.accuracy, 0.8);
  assert.notEqual(resolved.values.macro_f1, cvResult.metrics.fold_mean.test.macro_f1);
});

test('CV 的 Train/Valid 标量取 fold mean 并携带 fold std', () => {
  const train = resolveSplitScalars(cvResult, 'train');
  assert.equal(train.aggregation, 'fold_mean');
  assert.equal(train.values.accuracy, 0.9);
  assert.equal(train.foldStd.accuracy, 0.02);
});

test('holdout 全部使用 direct 口径', () => {
  const holdout = {
    evaluation: { strategy: 'stratified_holdout' },
    metrics: {
      splits: {
        train: { aggregation: 'direct', values: { accuracy: 0.95 } },
        test: { aggregation: 'direct', values: { accuracy: 0.9 } },
      },
    },
  };
  assert.equal(resolveSplitScalars(holdout, 'test').aggregation, 'direct');
  assert.equal(resolveSplitScalars(holdout, 'test').values.accuracy, 0.9);
  assert.equal(validateCvAggregation(holdout).ok, true);
});

test('CV 口径校验发现 test 误用 fold_mean，并回退到 pooled OOF', () => {
  const bad = JSON.parse(JSON.stringify(cvResult));
  bad.metrics.splits.test = { aggregation: 'fold_mean', values: { accuracy: 0.79 } };
  const check = validateCvAggregation(bad);
  assert.equal(check.ok, false);
  assert.ok(check.violations.some((item) => item.includes('pooled_oof')));
  const resolved = resolveSplitScalars(bad, 'test');
  assert.equal(resolved.aggregation, 'pooled_oof');
  assert.equal(resolved.values.macro_f1, 0.81);
});

test('CV 口径校验发现图表 train/valid 误用 pooled_oof', () => {
  const bad = JSON.parse(JSON.stringify(cvResult));
  bad.analysis.splits.train.aggregation = 'pooled_oof';
  const check = validateCvAggregation(bad);
  assert.equal(check.ok, false);
  assert.ok(check.violations.some((item) => item.includes('pooled_cross_fold')));
});

// ---------- artifact 展示规则 ----------

const artifactFixtures = [
  { name: 'metrics.json', label: '总体指标', downloadable: true, exists: true, integrity: 'ok', size_bytes: 128, download_url: '/api/training/runs/abc/artifact/metrics.json' },
  { name: 'model.pkl', downloadable: false, reason: '模型对象不开放下载' },
  { name: 'scaler.joblib', downloadable: false, reason: '内部拟合对象不开放下载' },
  { name: 'status.json', downloadable: false, reason: '易变投影不开放下载' },
  { name: 'manifest.json', downloadable: false, reason: '内部索引不开放下载' },
  { name: 'history.csv', downloadable: false, reason: '训练尚未成功完成，结果文件不可下载', exists: false, integrity: 'missing', download_url: null },
  { name: 'evil.json', downloadable: true, exists: true, integrity: 'ok', download_url: 'https://evil.example/x.json' },
];

test('黑名单 artifact 不进入展示列表', () => {
  const visible = visibleArtifacts(artifactFixtures);
  assert.deepEqual(visible.map((item) => item.name), ['metrics.json', 'history.csv', 'evil.json']);
  for (const name of ['model.pkl', 'model.pt', 'status.json', 'manifest.json', 'x.joblib']) {
    assert.ok(isForbiddenArtifactName(name), name);
  }
  assert.ok(!isForbiddenArtifactName('metrics.json'));
});

test('downloadable=true 且 URL 安全才允许下载', () => {
  const ok = artifactActionState(artifactFixtures[0]);
  assert.equal(ok.enabled, true);
  const history = artifactActionState(artifactFixtures[5]);
  assert.equal(history.enabled, false);
  assert.ok(history.reason.includes('不可下载'));
  const evil = artifactActionState(artifactFixtures[6]);
  assert.equal(evil.enabled, false);
  assert.ok(evil.reason.includes('不安全'));
});

test('下载 URL 安全校验', () => {
  assert.ok(isSafeDownloadUrl('/api/training/runs/abc/artifact/metrics.json'));
  assert.ok(isSafeDownloadUrl('/api/files?path=x'));
  assert.ok(!isSafeDownloadUrl('https://evil.example/api/x'));
  assert.ok(!isSafeDownloadUrl('http://evil.example/x'));
  assert.ok(!isSafeDownloadUrl('//evil.example/x'));
  assert.ok(!isSafeDownloadUrl('/api/../storage/x'));
  assert.ok(!isSafeDownloadUrl('C:\\data\\x.csv'));
  assert.ok(!isSafeDownloadUrl(''));
  assert.ok(!isSafeDownloadUrl(null));
});

// ---------- 服务器路径检查 ----------

test('干净的 run-result 载荷不报告服务器路径', () => {
  const clean = {
    schema_version: 'run-result-v1',
    dataset: { name: 'teacher.csv', sha256: 'ab12', curve_count: 90 },
    artifacts: [{ name: 'metrics.json', download_url: '/api/training/runs/abc/artifact/metrics.json', sha256: 'ff' }],
    analysis: { splits: { test: { confusion_matrix: [[1]] } } },
  };
  assert.deepEqual(findServerPaths(clean), []);
});

test('检出 data_path 键与绝对路径值', () => {
  const dirty = { config: { data_path: 'D:\\data\\a.csv' } };
  assert.deepEqual(findServerPaths(dirty), ['config.data_path']);
  assert.deepEqual(findServerPaths({ note: 'C:/Users/lenovo/x' }), ['note']);
  assert.deepEqual(findServerPaths({ log: '/home/ubuntu/runs/1' }), ['log']);
  assert.deepEqual(findServerPaths({ nested: [{ run_dir: '/var/lib/x' }] }), ['nested[0].run_dir']);
  assert.ok(looksLikeServerPath('D:\\x\\y.csv'));
  assert.ok(!looksLikeServerPath('/api/training/runs/abc/artifact/metrics.json'));
});

test('download_url 查询串按整体 URL 处理，不误报', () => {
  const payload = { download_url: '/api/files?path=D:\\preprocessed\\a.csv' };
  assert.deepEqual(findServerPaths(payload), []);
  assert.deepEqual(findServerPaths({ output_path: 'D:\\preprocessed\\a.csv' }), ['output_path']);
});

// ---------- 轮询器 ----------

test('poller 终态自动停止且不残留计时器', async () => {
  const scheduler = fakeScheduler();
  let calls = 0;
  const poller = createPoller({
    task: async () => { calls += 1; return { state: 'succeeded' }; },
    shouldContinue: (data) => data.state === 'running',
    setTimeoutFn: scheduler.set,
    clearTimeoutFn: scheduler.clear,
  });
  poller.start();
  await flush();
  assert.equal(calls, 1);
  assert.equal(poller.isRunning, false);
  assert.equal(scheduler.size, 0);
});

test('poller single-flight：进行中的请求不重复发起', async () => {
  const scheduler = fakeScheduler();
  let calls = 0;
  let release;
  const poller = createPoller({
    task: () => { calls += 1; return new Promise((resolve) => { release = resolve; }); },
    shouldContinue: () => true,
    setTimeoutFn: scheduler.set,
    clearTimeoutFn: scheduler.clear,
  });
  poller.start();
  assert.equal(calls, 1);
  poller.tick();
  poller.tick();
  assert.equal(calls, 1);
  assert.equal(scheduler.size, 0);
  release({ state: 'running' });
  await flush();
  assert.equal(scheduler.size, 1);
  poller.stop();
  assert.equal(scheduler.size, 0);
});

test('poller stop() 会 abort 进行中的请求，晚到结果被忽略', async () => {
  const scheduler = fakeScheduler();
  let signal;
  let release;
  let results = 0;
  const poller = createPoller({
    task: ({ signal: s }) => { signal = s; return new Promise((resolve) => { release = resolve; }); },
    onResult: () => { results += 1; },
    setTimeoutFn: scheduler.set,
    clearTimeoutFn: scheduler.clear,
  });
  poller.start();
  poller.stop();
  assert.equal(signal.aborted, true);
  assert.equal(poller.isRunning, false);
  release({ state: 'running' });
  await flush();
  assert.equal(results, 0);
  assert.equal(scheduler.size, 0);
});

test('poller 错误由 onError 决定是否继续', async () => {
  const scheduler = fakeScheduler();
  let attempts = 0;
  const poller = createPoller({
    task: async () => { attempts += 1; throw new Error('网络失败'); },
    onError: () => true,
    setTimeoutFn: scheduler.set,
    clearTimeoutFn: scheduler.clear,
  });
  poller.start();
  await flush();
  assert.equal(attempts, 1);
  assert.equal(poller.isRunning, true);
  assert.equal(scheduler.size, 1);
  scheduler.runNext();
  await flush();
  assert.equal(attempts, 2);
  poller.stop();

  const scheduler2 = fakeScheduler();
  const stopper = createPoller({
    task: async () => { throw new Error('boom'); },
    onError: () => false,
    setTimeoutFn: scheduler2.set,
    clearTimeoutFn: scheduler2.clear,
  });
  stopper.start();
  await flush();
  assert.equal(stopper.isRunning, false);
  assert.equal(scheduler2.size, 0);
});

// ---------- 图表纯函数 ----------

test('decimateSeries 小数据原样、长数据抽稀且不改入参', () => {
  const xs = [1, 2, 3, 4, 5];
  const ys = [5, 4, 3, 2, 1];
  const same = decimateSeries(xs, ys, 100);
  assert.deepEqual(same.xs, xs);
  assert.deepEqual(same.ys, ys);

  const longXs = Array.from({ length: 10000 }, (_, index) => index);
  const longYs = longXs.map((value) => Math.sin(value / 50));
  const decimated = decimateSeries(longXs, longYs, 500);
  assert.ok(decimated.xs.length <= 500);
  assert.equal(decimated.xs[0], 0);
  assert.equal(decimated.xs[decimated.xs.length - 1], 9999);
  assert.equal(longXs.length, 10000);
  assert.throws(() => decimateSeries([1], [1, 2], 10), /长度不一致/);
});

test('niceTicks 单调且覆盖范围', () => {
  const ticks = niceTicks(0, 10, 5);
  assert.ok(ticks.length >= 2);
  assert.ok(ticks[0] <= 0.0001 || ticks[0] <= 0);
  for (let index = 1; index < ticks.length; index += 1) {
    assert.ok(ticks[index] > ticks[index - 1]);
  }
  assert.deepEqual(niceTicks(3, 3), [3]);
  assert.deepEqual(niceTicks(Number.NaN, 1), []);
});

test('escapeHtml 覆盖五个危险字符', () => {
  assert.equal(escapeHtml(`<a href="x">'&`), '&lt;a href=&quot;x&quot;&gt;&#39;&amp;');
  assert.equal(escapeHtml(null), '');
});

test('混淆矩阵纯函数：清洗、强度与合计', () => {
  const matrix = normalizeMatrix([[3, 1], [0, '2'], 'junk']);
  assert.deepEqual(matrix, [[3, 1], [0, 2]]);
  assert.equal(cellIntensity(3, 3), 1);
  assert.equal(cellIntensity(0, 3), 0);
  assert.equal(cellIntensity(5, 0), 0);
  assert.deepEqual(matrixTotals(matrix), { total: 6, correct: 5 });
});

test('单样品解释优先使用样品谱线和样品 X 轴', () => {
  const resolved = resolveSampleSeries(
    { curve: [8, 9], sample_x_axis: [100, 200] },
    { baseline_curve: [1, 2], x_axis: [10, 20] },
  );
  assert.deepEqual(resolved, { curve: [8, 9], xAxis: [100, 200] });

  const legacy = resolveSampleSeries({}, { baseline_curve: [1, 2], x_axis: [10, 20] });
  assert.deepEqual(legacy, { curve: [1, 2], xAxis: [10, 20] });
});

test('格式化辅助函数', () => {
  assert.equal(formatMetric(0.812345678), '0.8123');
  assert.equal(formatMetric(null), '—');
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(2048), '2.0 KB');
  assert.equal(formatDuration(45.2), '45.2 秒');
  assert.equal(formatDuration(125), '2 分 5 秒');
  assert.equal(formatDuration(null), '—');
});

// ---------- 运行 ----------

let failed = 0;
for (const [name, fn] of tests) {
  try {
    await fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    failed += 1;
    console.error(`FAIL - ${name}`);
    console.error(error);
  }
}
if (failed) {
  console.error(`\n${failed}/${tests.length} 个测试失败`);
  process.exit(1);
}
console.log(`\n${tests.length} 个测试全部通过`);
