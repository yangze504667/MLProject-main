# 线束间解干涉：学生数据与辅助工具

本包提供 30 条聚类线束、统一宽度、箱体表面体素，以及干涉查询、效果评估和可视化工具。主要任务是降低线束之间的干涉长度，箱体作为辅助指标。可以采用分段优化或联合优化，结合 3–4 人的小组分工选择实现方式。任务与评定标准见 `作业说明.md`，接口与提交格式见 `框架说明.md`。

## 快速开始

解压后进入本目录，安装依赖：

```powershell
python -m pip install -r requirements.txt
python main.py --bundle 31
python view_bundles.py
python evaluate.py --output baseline_check.json
```

`main.py` 读取输入、显示原始线束间干涉长度并调用 `optimize.py` 的优化函数。首次运行时，待实现函数会抛出 `NotImplementedError` 并给出提示。完成函数后，运行同一命令即可保存候选路径及全场景评估报告；候选路径可以保留剩余干涉，效果由评估工具量化。

```powershell
python main.py --bundle 31 --output solution.json
python evaluate.py --solution solution.json --output evaluation.json
```

联合优化入口为 `optimize_scene`，返回 `{线束ID: 候选路径数组}`。实现后运行 `python main.py --joint --output solution.json`；两个优化接口均由小组实现。分段优化需要把各段整合为完整线束路径再提交。

查看器采用 Open3D 胶囊体方案：每段中心线生成圆柱，每个顶点生成球体，使用深色背景、法向光照和按线束 ID 确定的颜色。默认显示全部 30 条线束，半径与输入一致。按住鼠标拖动旋转，滚轮缩放。

修改 `view_bundles.py` 文件头部的 `SOLUTION_FILE = ROOT / "solution.json"`，再次运行会打开原始和提交结果两个窗口，并同步旋转、平移和缩放视角。没有提交文件时保持 `SOLUTION_FILE = None`。`BUNDLE_COUNT`、`BUNDLE_IDS` 控制数量与 ID；`SHOW_VOXELS` 控制箱体显示，默认为 False，便于观察线束间干涉。

按 `S` 保存截图：单窗口保存 `preview.png`，对比窗口保存 `comparison_before.png` 和 `comparison_after.png`。按 `C` 保存当前相机到 `view_camera.json`，下次启动自动读取。`visualize.py` 是同一查看器的兼容入口，参数统一在 `view_bundles.py` 中修改。

## 文件

| 文件 | 用途 |
|---|---|
| `data/bundles.json` | 30 条聚类线束，记录原 ID、中心折线和全局统一宽度 |
| `data/scene.json`、`data/occupancy.npy` | 箱体表面体素及坐标元数据 |
| `data/baseline_report.json` | 原始输入的逐线束与全场景干涉统计 |
| `main.py` | 数据读取、单条优化和联合优化入口 |
| `interference.py` | 空间索引、线束间连续干涉长度与评估，箱体辅助查询 |
| `optimize.py` | 路径校验及需要学生实现的空优化接口 |
| `evaluate.py` | 独立评估单条或多条提交，输出 JSON 报告 |
| `view_bundles.py` | Open3D 胶囊体查看器，支持原始或同步前后对比 |
| `visualize.py` | 调用同一 Open3D 查看器的兼容入口 |

统一半径为 3.2224061074 mm，直径为 6.4448122148 mm。主指标的原始线束间总干涉长度约 2733.328340 mm，29 条线束涉及干涉，共 71 对干涉关系；箱体辅助干涉长度约 882.506349 mm，涉及 19 条线束。ID 不连续，勿以列表下标代替 ID。

报告各字段和提交格式见 `框架说明.md`，小组任务边界见 `作业说明.md`。`evaluate.py` 的运行时间只代表评估耗时，算法搜索耗时须自行记录。胶囊网格和表面点云用于查看，最终几何判定以连续查询接口为准。
