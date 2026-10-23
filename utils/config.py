import argparse
import copy
import os

import yaml


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Read a YAML file
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base = cfg.pop("base", None)
    if base:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base)
        cfg = deep_merge(load_yaml(base_path), cfg)
    return cfg


# Parse a command-line value with YAML rules, also accepting numbers such as 2e-4
def parse_value(text):
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


# Apply overrides in place and return the config
def apply_overrides(cfg, overrides):
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parse_value(text)
    return cfg


# Argument parser
def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="extend", nargs="+", default=[])
    parser.add_argument("--fresh", action="store_true")
    return parser


# Resolve the final config: stage file (with its base), then --extra files, then --set overrides
def load_config(args):
    cfg = load_yaml(args.config)
    for extra in args.extra:
        cfg = deep_merge(cfg, load_yaml(extra))
    return apply_overrides(cfg, args.overrides)


# Return the output folder of one stage, creating it if needed
def stage_dir(cfg, name):
    path = os.path.join(cfg["paths"]["output_root"], name)
    os.makedirs(path, exist_ok=True)
    return path


# Path of a file written by an earlier stage
def stage_file(cfg, name, filename):
    return os.path.join(cfg["paths"]["output_root"], name, filename)


# Write the resolved config next to a stage's outputs
def save_config(cfg, out_dir):
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
