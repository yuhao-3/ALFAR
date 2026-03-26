# Baseline Methods Beyond ALFAR

This document lists baseline methods that are available in this repository (beyond ALFAR itself), plus references to any baseline methods that are only mentioned but not implemented here.

**Available Baselines**

- Visual Contrastive Decoding (VCD)
  - Implementation: `experiments/eval/vcd_sample.py`
  - How it is enabled: `evolve_vcd_sampling()` monkey-patches `transformers.generation.utils.GenerationMixin.sample`, and is called at the top of every `experiments/eval/alfar_*.py` script.
  - How it is used: the generation call passes `images_cd` and `cd_beta` to trigger contrastive decoding in `sample()`.
  - Where to look: `experiments/eval/vcd_sample.py`, `experiments/eval/alfar_free_llava.py`, `experiments/eval/alfar_mc_llava.py`, `experiments/eval/alfar_okvqa_llava.py`, `experiments/eval/alfar_evqa_llava.py`, plus the MiniGPT-4/Shikra/InstructBLIP variants.

- TCVM (Token-Level Causal Visual Masking) utilities
  - Implementation: `experiments/eval/tcvm_utils.py`
  - Status: utilities are present and imported by `experiments/eval/vcd_sample.py`, but they are not called anywhere in the current sampling logic. There are no scripts in `experiments/eval/` that expose TCVM as a runnable baseline.

**Mentioned But Not Implemented Here**

- PAI
  - Status: referenced in `README.md` as related prior work, but there is no PAI implementation or evaluation script in this repo.

**Notes**

- There are no standalone baseline evaluation scripts under `experiments/eval/` (everything is `alfar_*.py`). To run a non-ALFAR baseline, you would need to adjust those scripts to disable ALFAR-specific behavior and keep only the baseline logic you want (for example, keep VCD but remove attention reallocation).
