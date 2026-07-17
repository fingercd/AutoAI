/**
 * 串行轮询器：single-flight + AbortController。
 * - 任意时刻最多一个 task 在执行；上一次未完成不会发起下一次。
 * - stop() 会取消计时器并 abort 进行中的请求；晚到的结果被忽略。
 * - shouldContinue(data) 返回 false（如 Run 进入终态）时自动停止。
 * - onError(error) 返回 true 表示出错后继续轮询，否则停止。
 * 计时器可注入，便于 Node 测试。
 */
export function createPoller({
  task,
  intervalMs = 3000,
  shouldContinue = () => true,
  onResult = () => {},
  onError = () => {},
  setTimeoutFn = (fn, ms) => setTimeout(fn, ms),
  clearTimeoutFn = (id) => clearTimeout(id),
  abortControllerFactory = () => new AbortController(),
} = {}) {
  if (typeof task !== 'function') throw new TypeError('poller 需要 task 函数');
  let timer = null;
  let inFlight = null;
  let running = false;
  let generation = 0;

  function schedule() {
    if (!running) return;
    timer = setTimeoutFn(tick, intervalMs);
  }

  async function tick() {
    timer = null;
    if (!running || inFlight) return;
    const gen = generation;
    const controller = abortControllerFactory();
    inFlight = controller;
    try {
      const data = await task({ signal: controller.signal });
      if (!running || gen !== generation) return;
      inFlight = null;
      onResult(data);
      if (!running || gen !== generation) return;
      if (shouldContinue(data)) {
        schedule();
      } else {
        stop('terminal');
      }
    } catch (error) {
      if (!running || gen !== generation) return;
      inFlight = null;
      if (controller.signal.aborted || error?.name === 'AbortError') return;
      const keepGoing = onError(error) === true;
      if (keepGoing && running) {
        schedule();
      } else {
        stop('error');
      }
    }
  }

  function start() {
    if (running) return;
    running = true;
    tick();
  }

  function stop() {
    if (!running && !inFlight) return;
    running = false;
    generation += 1;
    if (timer !== null) {
      clearTimeoutFn(timer);
      timer = null;
    }
    if (inFlight) {
      try {
        inFlight.abort();
      } catch (_) {
        /* abort 失败不阻断停止 */
      }
      inFlight = null;
    }
  }

  return {
    start,
    stop,
    tick,
    get isRunning() {
      return running;
    },
    get isInFlight() {
      return inFlight !== null;
    },
  };
}
