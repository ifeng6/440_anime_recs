import argparse


_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(model):
    if model not in _funcs:
        raise ValueError(f"unknown model {model}")
    return _funcs[model]()


def main():
    parser = argparse.ArgumentParser()
    models = sorted(_funcs.keys())
    parser.add_argument(
        "models",
        choices=(models + ["all"]),
        nargs="+",
        help="A model to run, or 'all'.",
    )
    args = parser.parse_args()
    for q in args.models:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run(q)

        else:
            run(q)