import argparse
import json
from pathlib import Path

simulator_parameters = {
    "simulation": "debug_config",
    "evaluation": "S1",
    "city": "nyc",
    "horizon": "horizon_1_2k",
    "fleet": "fleet_20",
    "infrastructure": "S05_1_4",
}

extra_parameters = {
    "agent": "nearest",
}

# Mapping of parameter names to their subfolder in config directory
config_folder_mapping = {
    "simulation": "simulation",
    "evaluation": "evaluation",
    "city": "city",
    "horizon": "horizon",
    "fleet": "fleet",
    "infrastructure": "infrastructure",
    "agent": "agent",
}


def get_parser(**kwargs):
    parser = argparse.ArgumentParser()
    merged_args = {**simulator_parameters, **extra_parameters, **kwargs}
    for key, value in merged_args.items():
        if isinstance(value, bool):
            arg_type = _parse_bool
        elif isinstance(value, (int, float)):
            arg_type = type(value)
        else:
            arg_type = str
        parser.add_argument(f"--{key}", type=arg_type, default=value)
    return parser


def _parse_bool(value):
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def load_config(
    args,
    source_dir: str | Path = "./data/config",
    *,
    data_root: str | Path | None = None,
):
    """Load named configurations and overrides.

    ``data_root`` optionally anchors relative city input and agent checkpoint
    paths. Without it, paths retain their historical working-directory meaning.
    """
    instance_config = {}
    extra_config = {}

    # process the main config files
    for key, value in vars(args).items():
        # at the moment, we only want to process the main config files
        if len(key.split("_")) != 1:
            continue

        # Skip if this parameter doesn't have a folder mapping
        if key not in config_folder_mapping:
            continue

        # load the config file from the appropriate subfolder
        subfolder = config_folder_mapping[key]
        config_loc = Path(source_dir) / subfolder / f"{value}.json"
        with open(config_loc, encoding="utf-8") as config_file:
            # place the config in the appropriate dictionary
            if key in simulator_parameters:
                instance_config[key] = json.load(config_file)
            else:
                extra_config[key] = json.load(config_file)

    # process the overrides and additional parameters
    for key, value in vars(args).items():
        split_key = key.split("_", 1)
        if len(split_key) != 2:
            continue
        if split_key[0] in instance_config:
            instance_config[split_key[0]][split_key[1]] = value
        elif split_key[0] in extra_config:
            extra_config[split_key[0]][split_key[1]] = value
        else:
            raise (ValueError(f"Unknown parameter key: {split_key[0]}"))

    if data_root is not None:
        root = Path(data_root).resolve()
        for config, fields in (
            (
                instance_config.get("city", {}),
                ("geojson_file", "parking_file", "request_file"),
            ),
            (extra_config.get("agent", {}), ("checkpoint_path",)),
        ):
            for field in fields:
                value = config.get(field)
                if value and not Path(value).is_absolute():
                    config[field] = str(root / value)
    return instance_config, extra_config
