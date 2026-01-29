# Computational Optimization
MATH 5001 Computational Optimization (Spring 2026) course repo

# lecture note
see the pdf file /lecture_notes/Computational_Optimization.pdf

# code

## install venv

in root dir:

- initialize: `python3.10 -m venv venv`

- activate: `. venv/bin/activate`

- install dependencies (including external llm's dependencies shown in its pyproject.toml):
    
`pip install pip --upgrade`

`pip install -r requirements-dev.txt`

`pre-commit install`

## run via CLI

in root dir, under venv:

- activations: `python -m src.main activations`

## (for students) how to contribute

you must checkout your own branch from main, and then propose a PR to main.
Your branch can be merged only after my review and approval.
