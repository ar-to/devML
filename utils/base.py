
import pathlib
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

def load_dataset(filename: str) -> pd.DataFrame:
  """Load the dataset from the specified CSV file.

  Note: CSV must be in UTF-8 encoding and contain a header row. The function reads the CSV file into a pandas DataFrame and performs initial preprocessing.

  Initial Preprocessing: Strips any leading or trailing whitespace from the column names to ensure they are clean and consistent.

  Parameters
  ----------
  filename : str
    The name of the CSV file containing the dataset. e.g., "Life_Expectancy_Data.csv".

  Returns
  -------
  pd.DataFrame
      A DataFrame containing data.
  """
  _DATA_PATH = pathlib.Path(filename)
  if not _DATA_PATH.exists():
      raise FileNotFoundError(
          f"{filename} is missing from the lab directory. Please download it or ask the TA "
          "for assistance."
      )


  df = pd.read_csv(filename)
  df.columns = df.columns.str.strip()
  return df