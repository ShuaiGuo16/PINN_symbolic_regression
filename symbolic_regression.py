"""
    Symbolic regression to discover analytical functional forms for trained neural network models

    Author: Shuai Guo (Shuaiguo0916@hotmail.com)
"""

from pysr import PySRRegressor
import pandas as pd
import numpy as np
from pathlib import Path

# Load dataset
cwd = Path(__file__).resolve().parent
df = pd.read_csv(cwd/'f_NN_IO.csv')

# Symbolic regression model
model = PySRRegressor(
    niterations=20,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    loss="L1DistLoss()",
    model_selection="score",
    complexity_of_operators={
        "sin": 3, "cos": 3, "exp": 3,
        "inv(x) = 1/x": 3
    }
)

# NN for f1
X = df.iloc[:, :4].to_numpy()
f1 = df.loc[:, 'f2'].to_numpy()

model.fit(X, f1)
