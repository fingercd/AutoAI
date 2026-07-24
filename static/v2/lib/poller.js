/**
 * poller.js —— 串行轮询器（single-flight + AbortController）。
 *
 * 模块职责：v2 前端用它周期性拉取 Run 状态 / 结果（例如训练记录页、
 * 结果页等待训练完成），直到 Run 进入终态（succeeded/failed/cancelled）
 * 或组件主动停止。
 *
 * 设计要点（为什么这样写）：
 * - 串行 single-flight：任意时刻最多一个 task 在执行，上一次请求未完成
 *   时到点也不会发起下一次，避免慢请求堆积导致状态乱序或压垮后端；
 * - AbortController：stop() 会同时取消计时器并 abort 进行中的 fetch，
 *   页面切换后不再持有悬挂请求；
 * - generation 代际标记：stop() 后晚到的响应/错误会被直接丢弃，
 *   防止"已停止的轮询又复活"或旧结果覆盖新状态；
 * - shouldContinue(data) 返回 false（如 Run 进入终态）时自动停止；
 * - onError(error) 返回 true 表示出错后继续轮询（容忍瞬时网络抖动），
 *   否则停止；
 * - 计时器与 AbortController 工厂均可注入，便于在 Node 下做单元测试
 *   （不依赖浏览器真实计时器）。
 *
 * @param {Object} options
 * @param {(ctx: {signal: AbortSignal}) => Promise<any>} options.task
 *        每轮执行的异步任务，必须尊重 signal（通常直接传给 fetch）。
 * @param {number} [options.intervalMs=3000] 两轮之间的间隔毫秒数。
 * @param {(data: any) => boolean} [options.shouldContinue]
 *        拿到结果后判断是否继续轮询；返回 false 自动 stop。
 * @param {(data: any) => void} [options.onResult] 每轮成功结果回调。
 * @param {(error: any) => (boolean|void)} [options.onError]
 *        错误回调；返回 true 表示忽略错误继续轮询，其他返回值停止。
 * @param {Function} [options.setTimeoutFn] 可注入的 setTimeout。
 * @param {Function} [options.clearTimeoutFn] 可注入的 clearTimeout。
 * @param {Function} [options.abortControllerFactory] 可注入的 AbortController 工厂。
 * @returns {{start: Function, stop: Function, tick: Function,
 *            isRunning: boolean, isInFlight: boolean}}
 * @throws {TypeError} task 不是函数时立即抛出（编程错误，尽早暴露）。
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
  // —— 内部状态 ——
  let timer = null;      // 当前挂起的计时器 id；null 表示没有已排程的下一轮
  let inFlight = null;   // 进行中请求的 AbortController；非 null 即 single-flight 占用
  let running = false;   // start() 之后为 true，stop() 后复位
  let generation = 0;    // 代际计数：stop() 时 +1，用于识别并丢弃"晚到"的旧轮结果

  /** 排程下一轮 tick；仅在运行中排程，停止后不再复活。 */
  function schedule() {
    if (!running) return;
    timer = setTimeoutFn(tick, intervalMs);
  }

  /**
   * 执行一轮轮询。
   * 关键防护顺序：
   * 1. 进入时若已停止或已有请求在飞，直接返回（保证串行）；
   * 2. 记下当前 generation，await 之后逐项比对——stop() 会把 generation +1，
   *    此后任何晚到的 resolve/reject 都被静默丢弃；
   * 3. onResult / shouldContinue 回调内部可能调用 stop()，所以回调返回后
   *    要再次检查 running 与 generation 再决定是否排程。
   */
  async function tick() {
    timer = null;
    if (!running || inFlight) return;
    const gen = generation;
    const controller = abortControllerFactory();
    inFlight = controller;
    try {
      const data = await task({ signal: controller.signal });
      if (!running || gen !== generation) return;   // 晚到的结果：丢弃
      inFlight = null;
      onResult(data);
      if (!running || gen !== generation) return;   // onResult 里可能已 stop
      if (shouldContinue(data)) {
        schedule();
      } else {
        stop('terminal');   // 业务终态（如 Run succeeded/failed/cancelled）自动停止
      }
    } catch (error) {
      if (!running || gen !== generation) return;   // 晚到的错误：丢弃
      inFlight = null;
      // 自己 abort 触发的 AbortError 不是真错误，直接忽略
      if (controller.signal.aborted || error?.name === 'AbortError') return;
      const keepGoing = onError(error) === true;    // 严格 === true 才算"继续"
      if (keepGoing && running) {
        schedule();
      } else {
        stop('error');
      }
    }
  }

  /** 启动轮询：幂等（重复调用不产生第二个循环），立即执行第一轮。 */
  function start() {
    if (running) return;
    running = true;
    tick();
  }

  /**
   * 停止轮询：复位 running、提升 generation（使晚到结果失效）、
   * 清掉已排程计时器，并 abort 进行中的请求。
   * abort 用 try/catch 包裹：即使底层实现抛错也不阻断停止流程。
   * 参数 _reason（'terminal'/'error'/手动）当前仅作语义标记，不影响行为。
   */
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
