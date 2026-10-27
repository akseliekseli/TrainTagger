from schema import And, Optional, Or, Schema


SIGNAL_PROCESS_SCHEMA = {
    Optional("name"): str,
    "input": Or(str, [str]),
    "output": str,
    Optional("tree"): str,
}


UNIFIED_CONFIG_SCHEMA = Schema(
    {
        "experiment": {
            "name": str,
            "output": str,
            "training_percentage": And(
                Or(int, float),
                lambda percentage: 0.0 < percentage <= 100.0,
            ),
        },
        "data_config": {
            "input": Or(str, [str]),
            "output": str,
            Optional("tree"): str,
            Optional("step_size"): str,
            Optional("ratio"): And(
                Or(int, float),
                lambda ratio: 0.0 < ratio <= 1.0,
            ),
            Optional("num_workers"): And(
                int,
                lambda num_workers: num_workers >= 1,
            ),
            Optional("extras"): str,
            Optional("signal_processes"): [SIGNAL_PROCESS_SCHEMA],
        },
    },
    ignore_extra_keys=True,
)
