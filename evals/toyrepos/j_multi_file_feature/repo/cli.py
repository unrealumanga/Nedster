import argparse

from format import format_item
from models import Item


def build_parser():
    parser = argparse.ArgumentParser(prog="inventory")
    parser.add_argument("--name", default="widget")
    parser.add_argument("--count", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    item = Item(args.name, args.count)
    print(format_item(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
