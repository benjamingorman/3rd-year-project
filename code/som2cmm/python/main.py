import argparse
import json
import logging
import numpy as np
import os
import os.path
import shutil
import toml

import som2cmm.encoding as encoding
import som2cmm.utils as utils
import som2cmm.cmm as cmm

def validate_config(config):
    assert("working_directory" in config)
    assert("input_patterns_file" in config
           and os.path.isfile(config["input_patterns_file"]))
    assert("encoding" in config)

def setup_logging():
    """Sets logging to log to stdout as well as a file"""
    log = logging.getLogger("som2cmm")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler("debug.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s]%(name)s:%(levelname)s: %(message)s", "%H:%M:%S"))
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)
    return log

def run_experiment(config):
    patterns = utils.load_patterns_file(config["input_patterns_file"])
    key_patterns   = [pair[0] for pair in patterns]
    value_patterns = [pair[1] for pair in patterns] 

    key_encoder_cfg = config["encoding"]["keys"]
    value_encoder_cfg = config["encoding"]["values"]

    if value_encoder_cfg.get("copy_keys_encoding", False):
        log.info("Encoding values the same as keys")
        value_encoder_cfg = key_encoder_cfg

    key_encoder   = encoding.get_scheme(key_patterns, key_encoder_cfg)
    value_encoder = encoding.get_scheme(value_patterns, value_encoder_cfg)

    key_patterns_enc   = key_encoder.encodeAll(key_patterns)
    value_patterns_enc = value_encoder.encodeAll(value_patterns)

    utils.save_patterns_file(key_patterns_enc, value_patterns_enc, "encoded_patterns.json")
    utils.create_cmm_input_file(key_patterns_enc, value_patterns_enc, "cmm_input.txt")

    recalled_patterns = cmm.run_experiment("cmm_input.txt", os.getcwd(),
            key_encoder.get_num_bits_in_encoding(), config["cmm"])
    # Represented as (n, 1) np ndarrays so convert
    recalled_patterns = [list(val[:, 0]) for val in recalled_patterns]

    decoded_recalled_patterns = value_encoder.decodeAll(recalled_patterns)

    results = list(zip(key_patterns, value_patterns, decoded_recalled_patterns))

    save_results_file("results.txt", results)
    save_stats_file("stats.json", results)

    log.info("DONE")

def save_results_file(output_path, results):
    """Saves the results to the given file.
    Args:
        output_path (str): The path to the output file
        results (tuple(list, list, list)): The list of results
    """
    with open(output_path, 'w') as f:
        for (i, (key_vec, data_vec, data_recalled)) in enumerate(results):
            f.write("key {}".format(i).ljust(9))
            f.write(str(key_vec))
            f.write("\n")
            f.write("original ")
            f.write(str(data_vec))
            f.write("\n")
            f.write("recalled ")
            f.write(str(data_recalled))
            f.write("\n\n")

def save_stats_file(output_path, results):
    """Saves statistics about the results to the given file
    Args:
        output_path (str): The path to the output file
        results (tuple(list, list, list) of results
    """
    stats = {}
    distances = []
    distance_pcts = []

    for (key_vec, data_vec, data_recalled) in results:
        key_vec = np.array(key_vec)
        data_vec = np.array(data_vec)
        data_recalled = np.array(data_recalled)

        dist = np.linalg.norm(data_vec - data_recalled)
        distances.append(dist)
        distance_pcts.append(dist / np.linalg.norm(data_vec))

    stats["mean distance"] = np.mean(distances)
    stats["mean distance %"] = np.mean(distance_pcts)

    with open(output_path, 'w') as f:
        f.write(json.dumps(stats, indent=4, sort_keys=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the experiment's config.toml file")
    args = parser.parse_args()

    assert(os.path.isfile(args.config_file))

    config = toml.load(args.config_file)
    validate_config(config)

    os.makedirs(config["working_directory"], exist_ok=True)
    shutil.copyfile(args.config_file,
                    os.path.join(config["working_directory"], "config.copy.toml"))
    os.chdir(config["working_directory"])

    log = setup_logging()

    log.info("RUNNING EXPERIMENT")
    log.info("Using config file: %s", args.config_file)
    log.info("Using working directory: %s", config["working_directory"])

    run_experiment(config)
