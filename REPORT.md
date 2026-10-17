# Report: design, model selection, results and limitations

All figures come from the final notebook run (version 1.2.0, Kaggle, two
Tesla T4 GPUs, four CPU cores, Python 3.13) unless marked otherwise. The raw
output is in `results/` and the hand review is in `review/review_results.xlsx`.

## 1. Design

### Pipeline

```
source -> preprocess -> static analysis -> LLM stage -> report
```

1. **Preprocess.** Parse with `ast`, tokenise with `tokenize`. If parsing
   fails, the syntax error becomes a finding (P001) and the line and token
   rules still run, so half-written code still gets feedback. Files are cut
   into windows of up to 200 lines along top-level definitions before they go
   to the model, so a function is never split.
2. **Static analysis.** 55 rules in four categories, plus metrics (cyclomatic
   complexity, nesting depth, comment ratio). Optionally ruff and bandit run as
   subprocesses; their findings are mapped onto the same categories and
   severities, and anything a built-in rule already reported is dropped.
3. **LLM stage.** The model receives the numbered source and the static
   findings as JSON. It explains each finding (naming the variable or function
   involved), proposes a fix, and may report logic errors the rules missed.
4. **Report.** Text, Markdown, JSON or SARIF. `--fail-on` gives CI an exit code.

### Why the model explains instead of hunting

Asking a small model to find every problem in a file from scratch produces
confident, plausible and frequently wrong findings. Giving it findings that
are already verified, and asking it to explain them, uses it for what it is
good at. Its own findings are a separate channel with three guards: a
confidence cut-off (default 0.6), deduplication against static findings, and
rejection of anything outside the reviewed window or in an unknown category.

### Integration choices

- **One file.** The whole tool is `code_review_assistant.py`, with the
  standard library as its only hard dependency. Everything else (models,
  ruff, bandit, gradio) is optional and loaded only when used.
- **Local only.** Backends are transformers (`hf`), Ollama, any
  OpenAI-compatible local server, and an offline stand-in for testing. Remote
  addresses are refused unless `--allow-remote` is passed.
- **Greedy decoding.** Temperature 0, so a file gets the same review on every
  run. Sampling at 0.1 made the 7B's results move by a case or two between
  identical runs, which made comparisons meaningless.
- **Failure is contained.** Unreadable model output, a dead server or a
  missing tool are recorded in the report; the static findings still come
  through.
- **Parallel workers.** Files and test cases are spread over processes. With
  `--llm hf`, each worker is pinned to one GPU and loads its own model copy,
  so two T4s review two files at once.

## 2. Model selection

Both candidates are Qwen2.5-Coder instruction models: trained on code, able to
follow a JSON output format, and small enough for free GPUs.

| | 3B (fp16) | 7B (4-bit NF4) |
| --- | --- | --- |
| GPU memory | about 6 GB | about 5 GB |
| Licence | Qwen Research License | Apache 2.0 |
| Logic bugs found (of 10) | 0 | 3 |
| Own findings that were right | n/a (none made) | 3 of 4 |
| Static findings explained | 32 of 35 | 35 of 35 |
| Explanations naming the actual code | 78% | 71% |
| Mean time per file | 4.4 s | 6.7 s |

**Recommendation: the 7B in 4-bit.** It is the only configuration that finds
anything the rules cannot, it explains every static finding, it is about 50%
slower per file, and it fits a 16 GB GPU with room to spare. It is also the
one with an open-source licence; the 3B's licence is research-only, which on
its own rules it out for the brief's open-source constraint.

The 3B never produced a finding of its own in any run. It is usable as an
explainer if speed matters more than coverage.

## 3. Results

### 3.1 Labelled corpus

40 files: 30 with defects the rules are built for, including clean controls
and edge cases (empty file, comments only, syntax error), and 10 with logic
errors no rule can see. 45 labels in total. A prediction matches a label if
the category is the same and it is within one line; each label can be
claimed once. On the 10 logic-bug cases a model finding must also name the
actual bug (a few keywords per case) to count.

| | Static only | 3B | 7B 4-bit |
| --- | ---: | ---: | ---: |
| Overall precision | 1.000 | 1.000 | 0.974 |
| Overall recall | 0.778 | 0.778 | 0.844 |
| Overall F1 | 0.875 | 0.875 | 0.905 |
| Rule-visible F1 (35 labels) | 1.000 | 1.000 | 1.000 |
| Rule-blind recall (10 labels) | 0.000 | 0.000 | 0.300 |

By category, static only: bug F1 0.722 (13 of 23 found, all 10 misses are
the rule-blind cases), security 1.000, style 1.000, optimisation 1.000.

**What the 7B said on the 10 logic bugs:**

| Case | Result |
| --- | --- |
| `off_by_one_sum` | Right: "Off-by-one error in loop range." |
| `empty_average` | Right: division by zero on an empty list |
| `early_return_search` | Right: `return None` inside the loop |
| `missing_increment` | Wrong: suggested `<=`; the real bug is that `i` never increases |
| the other six | Nothing reported |

The automatic scoring and a check by hand agree on all ten.

**How much to trust this.** The corpus was written by the same person as the
rules, so 1.000 on the rule-visible cases mostly shows that the rules do what
they were written to do. The 10 logic-bug cases are a small sample: one case
more or less moves recall by 10 points. Treat "3 of 10" as "a few", not as a
measured rate.

### 3.2 Confidence cut-off

In the final run all four of the 7B's own findings had confidence 0.9, so
every cut-off from 0.0 to 0.9 gives the same result (3 right, 1 wrong). In an
earlier run with the labels still visible, two extra findings at 0.5 were
split one right and one wrong, which is weak support for keeping the default
at 0.6. The evidence for any particular value is thin.

### 3.3 Real code: the standard library

The only realistic check available without a labelled real-world bug set.

| | |
| --- | --- |
| Files | 599, 302,717 lines |
| Crashes | 0 |
| Findings per 100 lines | 1.86 (2.06 before the rule fixes in 3.4) |
| Critical or high per 100 lines | 0.064 |

### 3.4 Precision on real code, checked by hand

50 findings were drawn from the standard library (about 12 per category) and
each was marked real (true and worth a reviewer's time) or false (wrong, or
the code is right as written).

**First sample:** 13 of 50 real (26%). Three rules caused 24 of the 37 false
alarms:

- unused variable, firing on tuple unpacking (`data, consumed = encode(s)`);
- double lookup, firing on `if k in d: d[k]`, an ordinary idiom;
- `assert`, firing on internal invariants, which is what `assert` is for.

These were narrowed: unpacking and loop targets are skipped, the double-lookup
rule became opt-in, and `assert` is flagged only when it checks one of the
function's own arguments. Findings across the standard library fell by 10%.

**Fresh sample, drawn after the fix** (so not tuned to): 17 of 50 real (34%).

| Category | First sample | Fresh sample |
| --- | ---: | ---: |
| bug | 2 / 13 | 3 / 12 |
| security | 4 / 12 | 5 / 12 |
| style | 4 / 13 | 3 / 13 |
| optimization | 3 / 12 | 6 / 13 |
| **all** | **13 / 50 (26%)** | **17 / 50 (34%)** |

The remaining false alarms are mostly four patterns: bare `except` that cleans
up and re-raises (6), `exec` or `shell=True` in functions whose purpose is to
run code (5), long lines in a generated file (5), and "use a list
comprehension" on a list that already holds items (5). Each has an obvious
fix, listed in section 5.

This is the most important number in the report. The standard library is
mature, heavily reviewed code, so it contains few real defects, and a rule
that fires there is more often describing a deliberate choice than a mistake.
Precision on a less polished codebase would likely be higher, but that was not
measured.

### 3.5 Usefulness of the model's advice

20 explanations and fixes from the 7B, optimisation suggestions first, scored
1 (wrong or harmful) to 5 (correct, specific, usable as written). The scores
were drafted with an AI assistant and then reviewed by the author; they are
not an independent expert panel.

| | Average |
| --- | ---: |
| All 20 | 3.55 |
| Optimisation suggestions (5) | 3.60 |
| The model's own findings (3) | 3.33 |

Five scored 5, six scored 4, six scored 3, one scored 2 and two scored 1. The
low scores share a pattern: the finding is right but the reasoning or the fix
is not. On an `assert (cond, "msg")` that can never fail, the model's fix was
to delete the assert. On MD5 for passwords, it said "passwords" and then
recommended SHA-256, which is also unsuitable. About a quarter of the
explanations are generic ("consider refactoring the function").

### 3.6 Speed

| | |
| --- | --- |
| Static analysis, one corpus file | about 0.8 ms |
| Static analysis, throughput | about 11,600 lines/s on one core |
| Four workers on four cores | 1.98x (identical findings) |
| 3B, per file | 4.4 s mean, 10.2 s p95 |
| 7B 4-bit, per file | 6.7 s mean, 11.7 s p95 |
| 7B, corpus of 40 files on two GPUs | 200 s wall clock, including model loading |
| Whole notebook | about 15 minutes |

The static figures come from a run where nothing else was using the CPU. In
the final run the model downloads competed for the CPU during the one-worker
benchmark, which inflated the speed-up to an impossible 8.8x on four cores;
that figure is not used.

Model latency rules out reviewing as you type, but is fine for a pull request
or a nightly batch job. The static layer alone is fast enough for an editor.

## 4. What evaluating this taught me

Four of the numbers above changed because the measurement was wrong, not the
tool. Each was caught by looking at what the model actually said instead of
trusting the score.

1. **The model could see the answers.** Each test case labels its bug with a
   `#@ bug` comment on the faulty line, and those comments were reaching the
   model. Stripping them dropped the 7B from 4 to 3 logic bugs found. A
   self-test now checks that no label reaches the reviewer.
2. **Line matching credited wrong answers.** A finding on the right line
   counted even when it described a different problem. Requiring the
   explanation to name the actual bug cut the 7B's apparent score from 6 to 4
   of 10 before the leak was found.
3. **The model's answers varied between runs.** Sampling at temperature 0.1
   moved results by a case or two. Decoding is now greedy.
4. **The notebook ran old code twice.** Kaggle keeps a notebook on the dataset
   version it was attached with, so uploading a new version did nothing. The
   notebook now checks the code's version and stops with instructions if it
   is stale.

The first published figure was "F1 1.000". The honest version is 0.905 on a
corpus written for the tool, 34% precision on real code, and 3 of 10 logic
bugs found by the model.

## 5. Limitations and future work

**Limitations**

- Single-file analysis: no cross-module call graph, no type inference, and
  SQL taint tracking follows one assignment.
- Python only. The pipeline, the LLM stage, the reporting and the evaluation
  are language-agnostic, but the preprocessing and rules are not.
- About two-thirds of findings on mature code are not worth acting on.
- The model misses most logic bugs and sometimes gives a wrong fix with
  confidence.
- The corpus is small and self-written; there is no external labelled bug set.
- Ratings were AI-drafted and author-reviewed, not given by independent
  experts.

**Next steps, in order of value**

1. Score against an external bug set (BugsInPy, or real bug-fix commits).
2. Fix the four remaining false-positive patterns: skip bare `except` that
   ends in `raise`; skip files marked as generated; treat `exec` inside
   functions whose job is executing code as informational; only suggest a
   comprehension when the list is created empty just before the loop.
3. Let users mark findings as wrong in the UI and feed that back into rule
   thresholds.
4. Show the model the surrounding module, not just the window, so it can see
   how a function is called.
5. Offer fixes as applyable patches, and add a second language.

## 6. Reproducing

```bash
python code_review_assistant.py selftest
python code_review_assistant.py eval -o eval_static.json
python code_review_assistant.py benchmark --workers 4 -o benchmark_results.json
python code_review_assistant.py eval --llm hf --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --quantize 4bit --workers 2 --gpus auto -o eval_llm_7b.json
```

Or run `Code_Review_Assistant.ipynb` on Kaggle with two T4 GPUs and internet
on; it does all of the above, both models, the audit sample, the rating sheet,
a batch demo and a UI check, in about 15 minutes.
