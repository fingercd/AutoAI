# 数据映射

首版支持 CSV/XLSX，工作表单元格是文本或静态数值。XLS 先另存 XLSX；含公式的表先保存为明确数值。CSV 默认 UTF-8，其他编码显式指定 `--encoding gbk`。Excel 文本 ID 保留前导零；数字单元格的显示格式不是存储值，样品 ID 依赖显示格式时先确认其真实文本编码。

输出为 `Index,Label,Sample_ID,Name,<真实坐标…>`。坐标是有限、唯一、严格递增的 float64 往返文本，最多 16,380 列；强度有限，格式转换保留原数值文本，不额外舍入。Label 按类别文本处理，Sample_ID 代表真实样品。当前后端要求各样品重复次数一致。

## 命令与映射

```text
python <skill>/scripts/prepare_dataset.py inspect source.xlsx
python <skill>/scripts/prepare_dataset.py convert --mapping mapping.json --output-dir outputs/experiment/prepared
```

所有位置都是 **1 基**，首尾包含；转置时，位置针对转置后的表。相对 source 路径相对于 mapping.json。输出目录必须为空。多行表头通过指定真实坐标所在 `header_row` 和首条测量 `data_start_row` 跳过说明行，不自动合并含义不明的表头。

```json
{
  "sort_coordinates": false,
  "tables": [
    {
      "source": "source.xlsx",
      "sheet": "光谱",
      "sha256": "由 inspect 获取的文件指纹",
      "transpose": false,
      "header_row": 2,
      "data_start_row": 3,
      "columns": {"label": 2, "sample_id": 3, "name": 4},
      "feature_start_column": 5
    }
  ]
}
```

`sheet` 对 CSV 使用 `csv`；XLSX 必须选择明确工作表。`data_end_row`、`feature_end_column` 可限定范围，默认读取到完整表末尾。完全空白行会跳过，实际输出行来源记录在 conversion.json。

`coordinates` 可提供与特征列逐点对应的真实坐标数组。`sort_coordinates=true` 同时重排坐标和强度；重复坐标仍拒绝。多个 tables 只在逐点同轴时合并，省略 index 映射会生成连续 Index。`label_value` 可用于已经明确“一张表一个类别”的数据。

缺少 Sample_ID 时先确认分组。仅用户确认每行独立后设置顶层 `independent_rows_confirmed=true`；不确定则保留待确认，不转换成训练数据。

## 预处理输出补齐元数据

原始曲线保留原文件名，预处理结果的 Name 是映射键：

```json
{
  "tables": [{
    "source": "preprocessed.csv",
    "columns": {"name": 4},
    "feature_start_column": 5
  }],
  "metadata_by_name": {
    "sample-A-1.txt": {"Label": "甲", "Sample_ID": "sample-A"},
    "sample-A-2.txt": {"Label": "甲", "Sample_ID": "sample-A"},
    "sample-B-1.txt": {"Label": "乙", "Sample_ID": "sample-B"},
    "sample-B-2.txt": {"Label": "乙", "Sample_ID": "sample-B"}
  }
}
```

同名原文件不能靠顺序猜测映射。真实测量缺失、标签冲突、重复次数不一致必须报告，不能复制、填零或删样品。普通年龄/性别/指标表不具备真实曲线坐标，不通过改列名适配。

输出包含 dataset.csv、mapping.json、conversion.json 和 originals/；其中转换报告保存文件指纹、行来源、操作与样品统计。服务端上传仍进行最终完整校验。
