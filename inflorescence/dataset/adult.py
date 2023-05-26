from typing import List, Optional

import pandas as pd

from .dataset import Dataset, download_openml


class AdultDataset(Dataset):
    """The UCI Adult dataset. A standard benchmark for ML
    fairness algorithms.
    https://archive.ics.uci.edu/ml/datasets/adult
    """

    def __init__(self, download: bool = True):
        self.openml_id = 1590
        self.cat_cols = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "capital-gain",
            "capital-loss",
        ]
        if download:
            self.dataset = self.download()
        else:
            self.dataset = pd.DataFrame()  # empty placeholder
        super().__init__()

    def download(self):
        dataset = download_openml(self.openml_id)[0].dropna()  # ID for Adult dataset
        dataset["class"] = dataset["class"] == ">50K"
        # Dichotomize country, US vs non-US
        dataset["native-country"] = dataset["native-country"].eq("United-States")
        # Dichotomized marital status, single = 0, married = 1
        dataset["marital-status"] = dataset["marital-status"].isin(
            ["Married-AF-spouse", " Married-civ-spouse"]
        )
        dataset["capital-gain"] = dataset["capital-gain"].gt(0)
        dataset["capital-loss"] = dataset["capital-loss"].gt(0)
        dataset = dataset.drop(["education-num", "fnlwgt"], axis=1)
        return dataset

    def cat_index(self, col):
        if col not in self.cat_cols:
            raise ValueError(f"Column {col} is not categorical.")
        return self.dataset[col].factorize(sort=True)[1]

    @property
    def columns(self):
        return self.dataset.columns

    # def __getitem__(self, item):
    #     try:
    #         if item in self.groups:
    #             return self.groups[item]
    #         return self.dataset[item]
    #     except ValueError:
    #         raise KeyError(item)

    def get_xy(self, group_var: Optional[str] = None):
        x = self.dataset.drop("class", axis=1)
        x = pd.get_dummies(x, columns=self.cat_cols).to_numpy().astype(float)
        if group_var:
            group = self.dataset[group_var].factorize(sort=True)[0]
            y = self.dataset[["class"]].assign(**{group_var: group}).to_numpy().astype(float)
        else:
            y = self.dataset["class"].to_numpy().astype(float)
        return x, y
