# Code Review Assistant

A code review tool for Python that combines static analysis with an
open-source code model running locally. It reports bugs, security issues,
style problems and places where the code could be faster, explains why each one
matters, and suggests a fix.

It works in two steps:

1. **Rules** (55 of them, plus ruff and bandit if installed) find problems and
   point to the exact line. This part is fast and always gives the same answer.
2. **A local model** (Qwen2.5-Coder) explains each finding in terms of the
   actual code, suggests a fix, and looks for logic bugs that no rule can catch,
   like a loop that skips the first item.

Nothing is sent to an online API. Everything runs on your own machine or GPU.

```
$ python code_review_assistant.py review samples/sql_injection.py

samples/sql_injection.py
------------------------
  CRIT     7  [security/S011] SQL query built by string interpolation
              | cursor.execute(query)
          why: Interpolating user input into SQL is the classic injection
               vector; the driver can escape values safely if you let it.
          fix: Use a parameterised query: cursor.execute(sql, (value,)).
```

## Results

These come from the final notebook run on Kaggle (two T4 GPUs, about 15
minutes). [REPORT.md](REPORT.md) goes through each number and its caveats.

**On my test set** (40 small files: 35 bugs the rules are built for, 10 logic
bugs no rule can see):

| | Rules only | + 3B model | + 7B model (4-bit) |
| --- | ---: | ---: | ---: |
| F1 score | 0.875 | 0.875 | **0.905** |
| Rule-detectable bugs found | 35 / 35 | 35 / 35 | 35 / 35 |
| Logic bugs found | 0 / 10 | 0 / 10 | **3 / 10** |
| Model's own findings that were right | - | - | 3 of 4 |
| Time per file | 0.8 ms | 4.4 s | 6.7 s |

**On real code I didn't write** (the Python standard library):

| | |
| --- | --- |
| Files scanned | 599 files, 302,717 lines, no crashes |
| Findings per 100 lines | 1.86 |
| Findings worth acting on (50 checked by hand) | 34% (it was 26% before I fixed three noisy rules) |
| Usefulness of the 7B's explanations (20 rated 1 to 5) | 3.55 on average |
| Speed of the rules | about 11,600 lines per second on one core, 2x faster on four |

In short: the rules are fast and reliable on what they were written for, but on
mature, well-reviewed code most of what they flag is a deliberate choice rather
than a mistake. The 7B model finds a few real bugs the rules miss; the 3B model
only adds explanations. That's why the 7B is the one I recommend.

## Setup

You need Python 3.10 or newer (I tested on 3.13). The rules only use the
standard library, so this is enough to start:

```bash
git clone https://github.com/hansineesoundararajan/code-review-assistant.git
cd code-review-assistant
python code_review_assistant.py selftest
```

For the UI, the model and the extra checkers:

```bash
pip install -r requirements.txt
```

### Running a model

With [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5-coder:7b
python code_review_assistant.py review src/ --llm ollama --model qwen2.5-coder:7b
```

With `transformers` on a GPU:

```bash
python code_review_assistant.py review src/ --llm hf --model Qwen/Qwen2.5-Coder-7B-Instruct --quantize 4bit
```

Any local server with an OpenAI-style API (vLLM, llama.cpp, LM Studio) also
works:

```bash
python code_review_assistant.py review src/ --llm openai --base-url http://localhost:8000/v1
```

Addresses outside your machine are refused unless you add `--allow-remote`.

## Usage

### Review files (batch mode)

```bash
python code_review_assistant.py review app.py
python code_review_assistant.py review src/ tests/ --exclude "migrations"
python code_review_assistant.py review src/ --format markdown -o review.md
python code_review_assistant.py review src/ --format sarif -o results.sarif
python code_review_assistant.py review src/ --categories security bug --min-severity medium
python code_review_assistant.py review src/ --external-tools                 # also run ruff and bandit
python code_review_assistant.py review src/ --llm hf --workers 2             # one model per GPU
```

### Paste code into a web page (interactive mode)

```bash
python code_review_assistant.py ui
```

Opens a Gradio page. Paste code, choose the categories and minimum severity, and
you get the review, the code metrics and how long it took.

### Use it in CI

`--fail-on high` makes the command fail when it finds something high or
critical, and the SARIF output shows up in GitHub's code scanning tab. The
workflow in [.github/workflows/code-review.yml](.github/workflows/code-review.yml)
runs the self-test, the evaluation and a review of this tool's own code on
every push.

### Check how well it works

```bash
python code_review_assistant.py selftest                 # 60 checks
python code_review_assistant.py eval --show-errors       # precision, recall and F1 on the test set
python code_review_assistant.py benchmark --workers 4    # findings and speed on the standard library
python code_review_assistant.py dump-corpus samples      # save the test files to a folder
```

## Running it on Kaggle (and why not Colab)

[Code_Review_Assistant.ipynb](Code_Review_Assistant.ipynb) runs everything:
both models, the scoring, the real-code check, the rating sheet, a batch
review and a UI check.

The brief asks for a Colab notebook, but I ran this on Kaggle instead. Kaggle's
free tier gives two T4 GPUs at once, so I could load one copy of the model on
each GPU and score both models in about half the time, while free Colab gives
one. Kaggle's *Save & Run All* also runs the whole notebook on their servers as
a saved version, so the run finishes even if my browser closes, and the log and
output files stay with that version. The notebook still works on Colab without
changes: it detects where it is running and clones the code from GitHub when
there is no Kaggle dataset.

To run it on Kaggle: set the accelerator to **GPU T4 x2**, turn internet on,
upload `code_review_assistant.py` as a dataset, and press **Save & Run All**.

## How it works

```
source code
  -> preprocess        parse with ast, read tokens, split long files into 200-line pieces
  -> static analysis   55 rules and code metrics (plus ruff and bandit if asked)
  -> model             explain each finding, suggest a fix, look for logic bugs
  -> report            text, Markdown, JSON or SARIF
```

**Preprocessing.** The file is parsed and tokenised. If it doesn't parse, the
syntax error becomes a finding and the line-based rules still run, so half-written
code still gets feedback. Long files are split along function and class
boundaries so the model never sees half a function.

**Rules.** 55 checks over the syntax tree and the tokens, plus metrics like
cyclomatic complexity and nesting depth. With `--external-tools`, ruff and
bandit run too, and anything that duplicates one of my rules is dropped.

**Model.** It gets the numbered code and the rule findings as JSON. It's asked
to explain each finding by naming the actual variable or function, suggest a
fix, and report logic bugs the rules missed. Its own findings are only kept if
its confidence is at least 0.6 (`--llm-confidence`). Decoding is greedy, so the
same file always gets the same review. If the model's reply can't be read, the
rule findings still come through and the problem is noted in the report.

**Backends.** `hf` (transformers, optional 4-bit), `ollama`, `openai` (any
OpenAI-style local server) and `offline`, a stand-in for testing without a
model.

**Running in parallel.** Files are shared out between worker processes. With
`--llm hf`, each worker gets its own GPU and its own copy of the model.

### What the rules check

| Category | Rules | Examples |
| --- | ---: | --- |
| Bug | 13 | mutable default arguments, bare `except`, changing a list while looping over it, duplicate dict keys, `return` inside `finally`, unreachable code, unused variables |
| Security | 13 | SQL injection (also through a variable), `shell=True`, `eval` and `exec`, pickle and YAML loading, hardcoded passwords and keys, MD5/SHA1, turned-off TLS checks |
| Style | 20 | PEP 8 names, long lines, wildcard and unused imports, missing docstrings, shadowing builtins, too many parameters, high complexity |
| Optimisation | 8 | building strings in a loop, `range(len(x))`, extra `.keys()`, building a list just to sum it, checking membership in a list |
| Parse | 1 | syntax errors |

`O008` (checking `k in d` and then reading `d[k]`) is off by default, because
when I checked it by hand it was never worth changing. Turn it on with
`--enable O008`.

To hide a finding on one line:

```python
legacy_call()  # noqa: B005
legacy_call()  # review: ignore[B005, Y014]
```

## Testing

- **Self-test**: 60 checks for the rules, the ruff and bandit parsers, the
  OpenAI-style backend (against a small local test server), GPU assignment in
  the worker pool, and the scoring code itself.
- **Test set**: 40 small files, each bug marked with a `#@ category` comment on
  its line. The comments are removed before review so the model can't see them.
  On the 10 logic-bug files, a model finding only counts if its explanation
  describes the actual bug, not just the right line.
- **Benchmark**: the whole standard library, to measure noise and speed.
- **Hand review**: [review/review_results.xlsx](review/review_results.xlsx)
  has 50 real-code findings marked real or false, 20 model explanations scored
  1 to 5, and what the model said on each logic bug.

## Files

| File | What's in it |
| --- | --- |
| `code_review_assistant.py` | The whole tool: rules, model stage, command line, UI and evaluation |
| `Code_Review_Assistant.ipynb` | The Kaggle notebook that runs everything |
| `REPORT.md` | Design, model choice, results and limitations |
| `results/` | Output files from the final Kaggle run |
| `review/` | The hand review (audit, ratings, model answers) |
| `.github/workflows/` | CI setup |

## Dependencies and licences

| Package | Licence | Needed for |
| --- | --- | --- |
| Python standard library | PSF | everything (the only thing you must have) |
| gradio | Apache 2.0 | the UI |
| transformers, accelerate | Apache 2.0 | `--llm hf` |
| torch | BSD-3-Clause | `--llm hf` |
| bitsandbytes | MIT | `--quantize 4bit` |
| ruff | MIT | `--external-tools` |
| bandit | Apache 2.0 | `--external-tools` |
| pylint (optional) | GPL-2.0 | `--tools pylint`, run as a separate program, never imported |
| Qwen2.5-Coder-7B-Instruct | Apache 2.0 | the model I recommend |
| Qwen2.5-Coder-3B-Instruct | Qwen Research License | only used for the size comparison |

## Limitations

- It looks at one file at a time, with no view of how functions are called
  from other modules, and no type inference.
- Python only.
- On mature code, about two thirds of findings aren't worth changing.
- The model misses most logic bugs and sometimes suggests a wrong fix
  confidently.
- The test set is small and I wrote it myself.

More detail, and what I'd do next, is in [REPORT.md](REPORT.md).

