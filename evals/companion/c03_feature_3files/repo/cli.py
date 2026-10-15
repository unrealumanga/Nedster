"""Command line front end: python cli.py [--db FILE] add|list|done ..."""

from __future__ import annotations

import argparse
import sys

from service import TaskService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="todo")
    parser.add_argument("--db", default="tasks.json", help="path of the task file")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add a task")
    add.add_argument("title")

    lst = sub.add_parser("list", help="list tasks")
    lst.add_argument("--open", action="store_true", help="hide finished tasks")

    done = sub.add_parser("done", help="mark a task finished")
    done.add_argument("id", type=int)
    return parser


def format_task(task) -> str:
    mark = "x" if task.done else " "
    return f"[{mark}] {task.id}: {task.title}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    service = TaskService(args.db)
    if args.command == "add":
        task = service.add(args.title)
        print(f"added {task.id}")
    elif args.command == "list":
        for task in service.list(include_done=not args.open):
            print(format_task(task))
    elif args.command == "done":
        try:
            service.complete(args.id)
        except KeyError as exc:
            print(exc.args[0], file=sys.stderr)
            return 1
        print(f"finished {args.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
