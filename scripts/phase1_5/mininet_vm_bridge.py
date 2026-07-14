#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import paramiko


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2222
DEFAULT_USER = "mininet"
DEFAULT_PASSWORD = "mininet"


def build_client(args: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        timeout=args.timeout,
    )
    return client


def cmd_exec(args: argparse.Namespace) -> int:
    client = build_client(args)
    try:
        _, stdout, stderr = client.exec_command(args.command, get_pty=args.pty, timeout=args.timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        status = int(stdout.channel.recv_exit_status())
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return status
    finally:
        client.close()


def cmd_put(args: argparse.Namespace) -> int:
    client = build_client(args)
    try:
        sftp = client.open_sftp()
        try:
            sftp.put(str(Path(args.local).resolve()), args.remote)
        finally:
            sftp.close()
        return 0
    finally:
        client.close()


def cmd_get(args: argparse.Namespace) -> int:
    client = build_client(args)
    try:
        local_path = Path(args.local).resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp = client.open_sftp()
        try:
            sftp.get(args.remote, str(local_path))
        finally:
            sftp.close()
        return 0
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small bridge for the Mininet VM used in FIX1 live reruns")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--username", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--timeout", type=int, default=120)
    subparsers = parser.add_subparsers(dest="action", required=True)

    exec_parser = subparsers.add_parser("exec")
    exec_parser.add_argument("command")
    exec_parser.add_argument("--pty", action="store_true")

    put_parser = subparsers.add_parser("put")
    put_parser.add_argument("local")
    put_parser.add_argument("remote")

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("remote")
    get_parser.add_argument("local")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "exec":
        raise SystemExit(cmd_exec(args))
    if args.action == "put":
        raise SystemExit(cmd_put(args))
    if args.action == "get":
        raise SystemExit(cmd_get(args))
    raise SystemExit(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()
