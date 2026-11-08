# devML
repo for ML related projects and tests

## Getting Started with a Python Model

> note some steps may not yet exist or be implemented (e.g. check results directory) but left here for reference as this repo is improved.

To get started with a Python machine learning model, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/devML.git
    cd devML
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Upgrade Packages
    ```bash
    pip install pip-review
    pip-review # similar to `pip list --outdated`
    pip-review --auto #upgrade all because pip is not easy for this
    pip freeze > requirements.txt # update
    ```
5. Install internal modules
Modules like `utils` are self contained modules that live within this project alone to help with helper methods like monitoring. This project follows a flat structure so it remains flexible to organizing. But custom modules like `utils` requires treating this project like a package and install it on the python interpreter local to this project via `venv` above. Simply run:
    ```bash
    pip install -e .
    ```
Restart the notebook kernel. Then open a file using it such as `explorations/utils_checks.ipynb` and run the cell that imports it to test.

### Running a Model
1. Prepare your dataset and place it in the `data/` directory.
2. Run the model script:
    ```bash
    python src/train_model.py
    ```
3. Check the output in the `results/` directory.

### Notes
- Modify `config.yaml` to adjust model parameters.
- Refer to the `docs/` folder for detailed explanations of each module.

## Knowledge Graph (graphify)
This repo has a [graphify](https://docs.graphify.com/open-source/quickstart) knowledge graph of its code, notebooks and docs. Claude Code and its subagents can query it with `/graphify query "<question>"` instead of re-reading the whole repo. Rebuild it with `/graphify .`, or run `/graphify . --update` to re-extract only changed files.
- Outputs are in `graphify-out/`. A log of each build is in [docs/graphify_changelog.md](docs/graphify_changelog.md).
- graphify can't read `.ipynb` files, so notebooks are converted to `.py` files in `graphify-out/notebooks/` before each build.
- **Git:** `.gitignore` excludes everything in `graphify-out/` except `graph.json` and `GRAPH_REPORT.md`. Those two files are committed so a fresh clone, or a subagent, can query the graph without rebuilding it. The cache, manifest, cost log, notebook conversions and `graph.html` stay local.
