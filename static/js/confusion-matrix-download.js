/** Shared, dependency-free confusion-matrix PNG renderer for both result UIs. */

export function normalizeConfusionMatrix(matrix) {
  if (!Array.isArray(matrix) || !matrix.length) return [];
  const size = matrix.length;
  return matrix.map((row) => Array.from({ length: size }, (_, index) => {
    const value = Number(Array.isArray(row) ? row[index] : 0);
    return Number.isFinite(value) && value >= 0 ? value : 0;
  }));
}

export function confusionMatrixPngFilename(runId, split) {
  const safeRunId = String(runId || 'unknown')
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48) || 'unknown';
  const safeSplit = ['train', 'valid', 'test'].includes(String(split)) ? String(split) : 'test';
  return `run_${safeRunId}__confusion_matrix_${safeSplit}.png`;
}

export function confusionMatrixCanvasLayout(matrix, labels = []) {
  const normalized = normalizeConfusionMatrix(matrix);
  const size = normalized.length;
  const safeLabels = Array.from({ length: size }, (_, index) => String(labels[index] ?? `class_${index}`));
  const longest = Math.max(1, ...safeLabels.map((label) => Array.from(label).length));
  const cellSize = Math.max(58, Math.min(82, 46 + longest * 2));
  const rowLabelWidth = Math.max(34, Math.min(220, 18 + longest * 12));
  const columnLabelHeight = Math.max(34, Math.min(116, 18 + longest * 6));
  const padding = 16;
  const axisTitleSize = 24;
  const legendHeight = 32;
  return {
    matrix: normalized,
    labels: safeLabels,
    size,
    cellSize,
    rowLabelWidth,
    padding,
    axisTitleSize,
    legendHeight,
    columnLabelHeight,
    gridX: padding + axisTitleSize + rowLabelWidth,
    gridY: padding + axisTitleSize + columnLabelHeight,
    width: padding * 2 + axisTitleSize + rowLabelWidth + size * cellSize,
    height: padding * 2 + axisTitleSize + columnLabelHeight + size * cellSize + legendHeight,
  };
}

function fitText(context, text, maxWidth) {
  const value = String(text);
  if (context.measureText(value).width <= maxWidth) return value;
  let output = value;
  while (output.length > 1 && context.measureText(`${output}…`).width > maxWidth) {
    output = output.slice(0, -1);
  }
  return `${output}…`;
}

function drawMatrix(context, layout) {
  const { matrix, labels, size, gridX, gridY, cellSize, rowLabelWidth } = layout;
  const maxValue = Math.max(0, ...matrix.flat());
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, layout.width, layout.height);
  context.fillStyle = '#334155';
  context.font = '600 13px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'top';
  context.fillText('预测类别', gridX + size * cellSize / 2, layout.padding);
  context.save();
  context.translate(layout.padding + 7, gridY + size * cellSize / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText('真实类别', 0, 0);
  context.restore();
  context.save();
  context.fillStyle = '#475569';
  context.font = '600 13px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
  context.textAlign = 'right';
  context.textBaseline = 'middle';
  for (let column = 0; column < size; column += 1) {
    const x = gridX + column * cellSize + cellSize / 2;
    const y = gridY - 12;
    context.save();
    context.translate(x, y);
    context.rotate(-Math.PI / 4);
    context.fillText(fitText(context, labels[column], layout.columnLabelHeight * 1.2), 0, 0);
    context.restore();
  }
  context.restore();

  context.strokeStyle = '#d8dee9';
  context.lineWidth = 1;
  for (let row = 0; row < size; row += 1) {
    context.fillStyle = '#475569';
    context.font = '600 13px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    context.fillText(
      fitText(context, labels[row], rowLabelWidth - 18),
      gridX - 12,
      gridY + row * cellSize + cellSize / 2,
    );
    for (let column = 0; column < size; column += 1) {
      const value = matrix[row][column];
      const intensity = maxValue > 0 ? Math.min(1, value / maxValue) : 0;
      const diagonal = row === column;
      const x = gridX + column * cellSize;
      const y = gridY + row * cellSize;
      context.fillStyle = diagonal
        ? `rgba(15, 118, 110, ${0.08 + intensity * 0.42})`
        : value > 0 ? `rgba(220, 38, 38, ${0.05 + intensity * 0.35})` : '#ffffff';
      context.fillRect(x, y, cellSize, cellSize);
      context.strokeRect(x, y, cellSize, cellSize);
      context.fillStyle = '#111827';
      context.font = '600 16px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(String(value), x + cellSize / 2, y + cellSize / 2);
    }
  }
  const legendY = gridY + size * cellSize + 18;
  const items = [
    { color: 'rgba(15, 118, 110, 0.50)', label: '预测正确' },
    { color: 'rgba(220, 38, 38, 0.40)', label: '误分类' },
  ];
  context.font = '12px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
  const itemWidths = items.map((item) => 12 + 6 + context.measureText(item.label).width);
  const groupWidth = itemWidths.reduce((sum, width) => sum + width, 0) + 20;
  let legendX = (layout.width - groupWidth) / 2;
  context.textAlign = 'left';
  context.textBaseline = 'middle';
  context.fillStyle = '#475569';
  items.forEach((item, index) => {
    context.fillStyle = item.color;
    context.fillRect(legendX, legendY - 6, 12, 12);
    context.strokeStyle = '#cbd5e1';
    context.strokeRect(legendX, legendY - 6, 12, 12);
    context.fillStyle = '#475569';
    context.fillText(item.label, legendX + 18, legendY);
    legendX += itemWidths[index] + 20;
  });
}

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('浏览器未能生成 PNG 文件'));
    }, 'image/png');
  });
}

export async function downloadConfusionMatrixPng({ matrix, labels = [], runId = '', split = 'test', aggregation = '' }) {
  const layout = confusionMatrixCanvasLayout(matrix, labels);
  if (!layout.size) throw new Error('当前分区没有可下载的混淆矩阵');
  if (typeof document === 'undefined') throw new Error('当前环境不支持浏览器 Canvas 下载');
  const scale = 2;
  const canvas = document.createElement('canvas');
  canvas.width = layout.width * scale;
  canvas.height = layout.height * scale;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('当前浏览器无法创建 Canvas 画布');
  context.scale(scale, scale);
  drawMatrix(context, layout);
  const blob = await canvasBlob(canvas);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = confusionMatrixPngFilename(runId, split);
  anchor.style.display = 'none';
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return anchor.download;
}
