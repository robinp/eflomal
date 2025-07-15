#!/bin/env sh

echo "Run eflomal-server in this directory, so it picks up server_config.json, then perform the sample request:"

curl -X POST $@ localhost:5000/api/align/v1 -H 'Content-type: application/json' -H 'Accept: application/json' -d '{"aligner": "my-align", "samplers": 3, "iters": {"1":64, "2": 32, "3": 8}, "trust_sents": false, "sents":[{"s":"The cow and grass", "t":"Die Kuh und das Gras"}]}'

# Approximate response:
# {"aligns":[{"fwd":"0-0 0-1 1-2 2-3 3-4","norm_score_fwd":3.954102087565899,"norm_score_rev":2.250345638880109,"rev":"0-0 1-1 2-3 3-4","score_fwd":5.56354,"score_rev":3.63664}]}
