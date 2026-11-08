import argparse
import json


def _build_parser():
    parser = argparse.ArgumentParser(prog="grosel")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="VRAM faz planı")
    plan.add_argument("--budget-gb", type=float, default=5.0)
    plan.add_argument("--width", type=int, default=1024)
    plan.add_argument("--height", type=int, default=576)
    plan.add_argument("--bits", type=int, default=4)
    plan.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    plan.add_argument("--json", action="store_true")

    solve = sub.add_parser("solve", help="Bütçe-zaman-kalite çözümü")
    solve.add_argument("--budget-gb", type=float, default=5.0)
    solve.add_argument("--max-time-s", type=float, default=8.0)
    solve.add_argument("--min-mp", type=float, default=0.45)
    solve.add_argument("--steps", type=int, choices=(3, 4, 6), default=4)
    solve.add_argument("--objective", choices=("time", "vram", "quality"), default="quality")
    solve.add_argument("--cold", action="store_true")
    solve.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="FastAPI studio")
    serve.add_argument("--port", type=int, default=7870)
    serve.add_argument("--host", default="127.0.0.1")
    return parser


def _format_plan(result):
    lines = [
        result["verdict"],
        (
            f"bütçe {result['budget_gb']:.2f} GB | tepe {result['peak_gb']:.3f} GB"
            f" | pay {result['headroom_gb']:.3f} GB"
        ),
    ]
    phases = " | ".join(f"{p['id']} {p['peak_gb']:.3f} GB" for p in result["phases"])
    lines.append(f"fazlar: {phases}")
    lines.append("seçenekler:")
    for option in result["options"]:
        state = "uygun" if option["feasible"] else "sığmıyor"
        support = "destekli" if option["supported"] else "deneysel"
        lines.append(
            f"  {option['bits']}-bit  tepe {option['peak_gb']:.3f} GB"
            f"  {state}  {support}"
        )
    lines.append(f"depolama çalışma {result['ram']['runtime_gb']:.2f} GB")
    return "\n".join(lines)


def _format_solve(result):
    lines = [f"durum: {result['status']}"]
    if result["status"] == "no-data" or result.get("chosen") is None:
        lines.append("not: depo telemetri (runs.json) içermiyor; seçilebilir plan yok.")
        return "\n".join(lines)
    chosen = result["chosen"]
    lines.append(
        f"plan: {chosen['quality']} / {chosen['steps']} adım / {chosen['resolution']}"
        f" / {chosen['native_mp']:.3f} MP / {chosen['total_time']:.2f} s"
        f" / {chosen['peak_gb']:.2f} GB"
    )
    binding = ", ".join(result.get("binding") or []) or "yok"
    lines.append(f"kısıtlar: {binding}")
    return "\n".join(lines)


def _cmd_plan(args):
    from web_app import resident_math

    result = resident_math.plan(
        args.budget_gb, args.width, args.height, args.bits, args.strict
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_plan(result))


def _cmd_solve(args):
    from web_app import inference_math

    result = inference_math.solve(
        args.budget_gb,
        args.max_time_s,
        args.min_mp,
        args.steps,
        warm=not args.cold,
        objective=args.objective,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_solve(result))


def _cmd_serve(args):
    import uvicorn

    from web_app import server

    uvicorn.run(server.app, host=args.host, port=args.port)


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        _cmd_plan(args)
    elif args.command == "solve":
        _cmd_solve(args)
    else:
        _cmd_serve(args)


if __name__ == "__main__":
    main()
