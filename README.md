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
    pip freeze requirements.txt
    ```

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
