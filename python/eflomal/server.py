from flask import Flask, request, make_response

import json
import os
import functools
import time
import math

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

        trust_sents = True
        if 'trust_sents' in req:
            f = req['trust_sents']
            if type(f) == bool:
                trust_sents = f
            else:
                raise InputFormatException("trust_sents should be bool")

        samplers = 3   # copied default
        if 'samplers' in req:
            samplers = int(req['samplers'])

        scoring = True
        if 'scoring' in req:
            f = req['scoring']
            if type(f) == bool:
                scoring = f
            else:
                raise InputFormatException("scoring should be bool")

        num_sents = len(req['sents'])
        sent_stoks = [0] * num_sents
        sent_ttoks = [0] * num_sents
        def input_iter(field, toks):
            for n, sent in enumerate(req['sents']):
                f = sent[field]
                if type(f) == list:
                    toks[n] = len(f)
                    f = ' '.join(f)
                elif type(f) == str:
                    toks[n] = len(f.split())
                else:
                    raise InputFormatException("Sentence should be string or list of strings")
                yield f
        src_iter = input_iter("s", sent_stoks)
        trg_iter = input_iter("t", sent_ttoks)

        t10 = time.time()
        with TemporaryDirectory() as td:
            fwd_fp = os.path.join(td, "req.fwd")
            rev_fp = os.path.join(td, "req.rev")
            fsc_fp = os.path.join(td, "rsc.fwd") if scoring else None
            rsc_fp = os.path.join(td, "rsc.rev") if scoring else None

            aligner.n_iterations = iters
            aligner.n_samplers = samplers
            try:
                aligner.align(src_iter, trg_iter,
                              links_filename_fwd=fwd_fp,
                              links_filename_rev=rev_fp,
                              scores_filename_fwd=fsc_fp,
                              scores_filename_rev=rsc_fp,
                              trust_sents=trust_sents,
                              quiet=log_level != "debug")
            except InputFormatException as e:
                return make_response(e.msg, 400)

            scores = []
            if scoring:
                with open(fsc_fp, 'r') as fscf, open(rsc_fp, 'r') as rscf:
                    for fs, rs in zip(fscf, rscf):
                        scores.append((float(fs), float(rs)))

            with open(fwd_fp, 'r') as fwdf, open(rev_fp, 'r') as revf:
                fr_pairs = []
                for n, (f, r) in enumerate(zip(fwdf, revf)):
                    res = { "fwd": f.strip(), "rev": r.strip() }
                    if scoring:
                        fs, rs = scores[n]
                        res["score_fwd"] = fs
                        res["score_rev"] = rs
                        res["norm_score_fwd"] = fs - math.log(sent_ttoks[n])
                        res["norm_score_rev"] = rs - math.log(sent_stoks[n])
                    fr_pairs.append(res)
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
