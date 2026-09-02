"""
List every registered feature extractor with its tunable hyperparameters.

    .venv/Scripts/python model/list_extractors.py

Use it to check that a newly added extractor was picked up, and to see the sweep
space it declares.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_extractor as fe


def main() -> None:
    names = fe.available()
    if not names:
        print("No extractors registered. Add one under model/extractors/.")
        return

    for name in names:
        cls = fe.get(name)
        space = cls.search_space()
        signature = inspect.signature(cls.__init__).parameters
        tunable = [
            key for key in signature
            if key not in {"self", "seq_len", "num_channels", "embedding_dim"}
        ]

        print(f"\n{name}  ({cls.__module__})")
        summary = (inspect.getdoc(cls) or "").strip().splitlines()
        if summary and not summary[0].startswith("Args:"):
            print(f"  {summary[0]}")

        for key in tunable:
            default = signature[key].default
            default = "required" if default is inspect.Parameter.empty else repr(default)
            values = space.get(key)
            sweep = f"  sweep: {values}" if values else ""
            print(f"    {key:<16} default={default:<10}{sweep}")

        combinations = 1
        for values in space.values():
            combinations *= len(values)
        if space:
            print(f"    -> {combinations} combinations in the declared search space")

    print(f"\n{len(names)} extractor(s) registered.")


if __name__ == "__main__":
    main()
