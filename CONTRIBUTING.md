# 参与开发

`src/smartcfd` 中的算法模块和 `tests/` 顶层六个算法测试模块均为占位文件，`configs` 为尚未接入代码的配置草案。已有公开数据处理脚本位于 `scripts/data/`，对应的 21 个回归测试位于 `tests/data/`；使用方式见 [公开数据说明](docs/public_data_v1.md)。

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

现有数据回归测试已接入 CI，无需联网或下载原始数据。也可使用 `python -m pytest tests/data -q` 执行。实现算法后，应补齐顶层预留测试并使用 `python -m pytest` 执行全部用例。

已有公开数据版本、来源记录与划分清单继续版本管理。新增原始数据、大型处理产物、模型权重和运行结果默认保留在本地；轻量数据划分清单可以放入 `data/splits` 并提交。实验配置、数据版本、随机种子与结果摘要应记录在 `docs/experiments.md`。第三方数据归属见 [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md)。

提交说明应写清改动目的、模块行为和验证方式。许可证尚未选择，请维护者先确定许可证并替换 `LICENSE` 占位文本，再发布带许可证的版本。
