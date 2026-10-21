# cuad/ — pipeline snapshot used for the CUAD intra-document experiment

The CUAD results in the paper (Fig. 5a and Supplementary Note 1: one Licensor–Licensee
software-licence agreement, Cap-on-Liability query, L2 versus MMR at int8, aggregate
statistics over 50 queries) were produced in August 2026 with the pipeline snapshot that
preceded the HotpotQA work and the later refactoring of the device path. This folder keeps
that snapshot's versions of the modules that changed afterwards, so the CUAD run can be
repeated exactly; every file **not** listed here was identical to its top-level counterpart
at the time of the CUAD run.

| File | Role in the CUAD run |
|---|---|
| `off_line_prepper.py` | Offline index of the contracts (clause-level chunking, multi-qa-mpnet-base-dot-v1, IVF/PQ) |
| `online_accelerator_device.py` | Retrieval with the device-calibrated distance models; intra-document `COMPARE` scope (L2 vs MMR within one contract) |
| `sweep_eval.py` | Sweep driver (λ, τ_dup, γ, top-k grid), writes `sweep_exports/<run_id>/retrieval_<run_id>.jsonl` |
| `llm_generator.py` | Llama-3-8B-Instruct generation at fp16/int8/int4 with the four-step legal prompt (Core Finding; Legal Context and Potential Risks; Executive Summary; Clause Reference Index) citing passages as "Chunk N" |
| `postprocess.py` | Local, offline resolution of "Chunk N" references to clause numbers, every substitution recorded |
| `run_eval_from_generation.py`, `evaluation/main.py`, `evaluation/data_loader.py`, `evaluation/utils/schema.py` | Evaluation of the normalised generations, including per-aspect retrieval coverage (NLI entailment of each retrieved chunk against the query's aspect queries) |
| `scripts_build_note_bundle.py`, `scripts_build_supplementary_note.py`, `build_note_docx.js`, `build_note_docx_en.js` | Assembly of Supplementary Note 1 (contract text, retrieved chunks, generated answers) |

The same CUAD functionality is also present in the current top-level modules, which
additionally support HotpotQA. To repeat the CUAD run as published, copy the files in this
folder over the top-level ones of the same name (preferably in a separate working copy),
prepare the CUAD index as described in the top-level README, and run
`sweep_eval.py --domain cuad ...` followed by generation and `run_eval_from_generation.py`.
