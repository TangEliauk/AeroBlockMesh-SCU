"""训练强化学习分块策略的命令行入口（待实现）。"""

import argparse


def main() -> None:
    """展示入口参数；算法接入前明确拒绝执行。"""
    parser = argparse.ArgumentParser(description="训练强化学习分块策略（项目骨架，功能待实现）")
    parser.add_argument(
        "--config",
        default="configs/rl.yaml",
        help="预留的配置文件路径（相对于仓库根目录）",
    )
    parser.parse_args()
    parser.error("当前仅提供项目骨架；训练强化学习分块策略功能尚未实现。")


if __name__ == "__main__":
    main()
