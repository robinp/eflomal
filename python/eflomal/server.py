from flask import Flask, request, make_response

import json
import os
import functools
import time

from eflomal import Aligner, sentences_from_joint_file
from tempfile import TemporaryDirectory

import logging
logger = logging.getLogger(__name__)


DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(process)d] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"

ACCEPT_LOG_LEVELS = ["error", "info", "debug"]


class InputFormatException(Exception):
    def __init__(self, msg):
        self.msg = msg


def create_app():
    app = Flask(__name__, instance_relative_config=True)  # why?

    app_config_path = os.environ.get('FLASK_APP_CONFIG')
    with open(app_config_path) as f:
        cfg = json.load(f)

    log_format = cfg.get("log_format", DEFAULT_LOG_FORMAT)

    log_level = cfg.get("log_level", "info")
    if not log_level in ACCEPT_LOG_LEVELS:
        raise Exception(f"log_level not one of {ACCEPT_LOG_LEVELS}")

    ll = None
    if log_level == "error":
        ll = logging.ERROR
    elif log_level == "info":
        ll = logging.INFO
    elif log_level == "debug":
        ll = logging.DEBUG

    logging.basicConfig( level=ll, format=log_format)

    logger.info("Read application config: %s", cfg)

    aligners = {}
    for acfg in cfg["aligners"]:
        name = acfg["name"]
        pri = acfg["priors"]
        logger.info(f"Loading aligner {name} with priors {pri}")
        aligners[name] = create_aligner(pri)

    @app.route('/api/align/v1', methods=['POST'])
    def alignV1():
        req = request.get_json()
        aligner = aligners[req['aligner']]

        iters = [32, 32, 32]
        if 'iters' in req and req['iters']:
            req_iters = req['iters']
            if "1" in req_iters: iters[0] = req_iters["1"]
            if "2" in req_iters: iters[1] = req_iters["2"]
            if "3" in req_iters: iters[2] = req_iters["3"]
        iters = tuple(iters)

        samplers = 3   # copied default
        if 'samplers' in req:
            samplers = int(req['samplers'])

        num_sents = len(req['sents'])
        def input_iter(field):
            for sent in req['sents']:
                f = sent[field]
                if type(f) == list:
                    f = ' '.join(f)
                if type(f) != str:
                    raise InputFormatException("Sentence should be string")
                yield f
        src_iter = input_iter("s")
        trg_iter = input_iter("t")

        t10 = time.time()
        with TemporaryDirectory() as td:
            fwd_fp = os.path.join(td, "req.fwd")
            rev_fp = os.path.join(td, "req.rev")

            aligner.n_iterations = iters
            aligner.n_samplers = samplers
            try:
                aligner.align(src_iter, trg_iter,
                              links_filename_fwd=fwd_fp,
                              links_filename_rev=rev_fp,
                              quiet=log_level != "debug")
            except InputFormatException as e:
                return make_response(e.msg, 400)

            with open(fwd_fp, 'r') as fwdf, open(rev_fp, 'r') as revf:
                fr_pairs = []
                for f, r in zip(fwdf, revf):
                    fr_pairs.append({ "fwd": f.strip(), "rev": r.strip() })
            if len(fr_pairs) != num_sents:
                raise Exception(f'Number of alignments differ from inputs: {len(fr_pairs)} != {num_sents}')
            res = { "aligns": fr_pairs }
            return res

    # Don't forget this.
    return app

def create_aligner(prior_path):
    # TODO(config) more config if needed
    aligner = Aligner()
    with open(prior_path, 'r', encoding='utf-8') as priors_input:
        aligner.preload_priors(priors_input)
    return aligner


def main():
    app = create_app()


if __name__ == '__main__':
    main()
