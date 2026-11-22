"""定位二维 SAC 网格演示。"""

import argparse
from pathlib import Path


DEMO_PAGE = Path(__file__).resolve().parents[1] / "experiments" / "sac_2d" / "demo" / "index.html"


def main() -> None:
    """打印离线演示入口，保留通用配置的待实现提示。"""
    parser = argparse.ArgumentParser(description="定位二维 SAC 网格离线演示")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="打印可在浏览器中打开的离线演示文件路径")
    source.add_argument(
        "--config",
        default="configs/meshing.yaml",
        help="预留的配置文件路径（相对于仓库根目录）",
    )
    args = parser.parse_args()
    if not args.demo:
        parser.error("通用网格可视化配置尚未接入。使用 --demo 查看二维 SAC 离线演示路径。")
    if not DEMO_PAGE.is_file():
        parser.error(f"找不到二维 SAC 离线演示文件 {DEMO_PAGE}")
    print(DEMO_PAGE)


if __name__ == "__main__":
    main()
