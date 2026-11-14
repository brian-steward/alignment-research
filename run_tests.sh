#!/bin/bash
# Runs every test suite, one process per suite.
# The two eval packages have modules with the same names (fingerprint, metrics, elicit, run), so they cannot
# share one pytest process. All suites run on CPU; none needs a GPU. Four tests (one in sw_program, three in
# coherence_pilot) load the GPT-2 tokenizer, which is downloaded from Hugging Face the first time (about 3 MB).
# Nothing else is downloaded.
cd "$(dirname "$(readlink -f "$0")")"
export CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false
status=0
for suite in evals/number_golf evals/coding_golf tests sw_program coherence_pilot s15_susceptibility; do
  echo "=== $suite ==="
  python -m pytest "$suite" -q --no-header -rs "$@" || status=1
done
exit $status
