#!/usr/bin/env python3
"""
Reproduce GitHub issue #5: simulate exactly what the Home Assistant
Wyoming integration does when you add the integration — using the SAME
official `wyoming` pip package HA uses.

HA flow: connect → send Describe → wait for Info (with wake programs).
If nothing valid comes back in time → "Failed to connect".

Usage: python repro_ha_client.py [--host 127.0.0.1] [--port 10400]
"""

import argparse
import asyncio

from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.info import Info


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10400)
    p.add_argument("--timeout", type=float, default=5.0)
    args = p.parse_args()

    print(f"[HA-sim] Connecting to tcp://{args.host}:{args.port} "
          f"(wyoming lib, same as HA integration)")
    try:
        async with AsyncTcpClient(args.host, args.port) as client:
            await client.write_event(Event(type="describe"))
            print("[HA-sim] → sent Describe, waiting for Info...")

            while True:
                event = await asyncio.wait_for(client.read_event(), args.timeout)
                if event is None:
                    print("[HA-sim] ✗ connection closed without Info")
                    print('[HA-sim] HA would show: "Failed to connect"')
                    return 1
                if event.type == "info":
                    info = Info.from_event(event)
                    progs = info.wake or []
                    if not progs:
                        print("[HA-sim] ✗ got Info but no wake programs — "
                              "HA cannot use this service")
                        return 1
                    print(f"[HA-sim] ✓ Info received, {len(progs)} wake program(s):")
                    for prog in progs:
                        for model in prog.models:
                            print(f"[HA-sim]    model: {model.name} "
                                  f"languages={model.languages} phrase={model.phrase!r}")
                    print("[HA-sim] SUCCESS — Home Assistant would connect")
                    return 0
                print(f"[HA-sim] ← unexpected event: {event.type}")
    except asyncio.TimeoutError:
        print(f"[HA-sim] ✗ TIMEOUT after {args.timeout}s — service never "
              f"answered Describe with a valid Info event")
        print('[HA-sim] HA would show: "Failed to connect"')
        return 1
    except (ConnectionRefusedError, OSError) as e:
        print(f"[HA-sim] ✗ connection error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
