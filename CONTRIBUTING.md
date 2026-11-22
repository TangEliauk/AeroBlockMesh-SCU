# 参与开发

`src/smartcfd` 包含二维 NACA 四块 O 型网格与 SAC 调参实现。其他研究模块和通用 YAML 保留为接口草案。公开数据处理脚本和其回归测试分别位于 `scripts/data/`、`tests/data/`。[公开数据说明](docs/public_data_v1.md)

使用 Python 3.10 或更高版本，在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

在对应模块中逐步添加实现；跨模块的数据约定先记录在 `docs/interfaces.md`，避免重复定义几何、分块与网格对象。算法功能应配套能够验证行为的测试，使用小型、来源清楚的测试数据。

提交前检查构建和语法：

```powershell
python -m build
python -m compileall -q src scripts tests
python -m unittest discover -s tests/data -p "test_*.py" -v
python scripts/generate_mesh.py --help
```

使用 `python -m pytest` 执行数据和算法测试。数据回归测试可离线运行。

已有公开数据版本、来源记录与划分清单继续版本管理。新增原始数据、大型处理产物和运行结果默认保留在本地。随包提供的二维演示检查点约 23 KB，可离线运行原型。实验配置、数据版本、随机种子与结果摘要见 [实验记录](docs/experiments.md)。第三方数据归属见 [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md)。

提交说明应写清改动目的、模块行为和验证方式。许可证尚未选择，请维护者先确定许可证并替换 `LICENSE` 占位文本，再发布带许可证的版本。
