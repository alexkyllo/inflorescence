import os
import platform
from io import StringIO
from typing import Optional, Tuple
from urllib.request import urlopen

import pandas as pd
from flwr.common.typing import NDArray

from .dataset import Dataset, download_openml

CACHE_PATH = (
    os.environ.get(
        "XDG_CACHE_HOME",
        os.path.join(
            "~",
            ".cache",
            "inflorescence",
        ),
    )
    if platform.system() == "Linux"
    else os.path.join("~", ".inflorescence")
)


def process(data):
    df = data.assign(c_charge_desc=data.c_charge_desc.fillna(""))
    df = df.assign(
        sex=df.sex.eq("Male"),
        c_charge_degree_felony=df.c_charge_degree.eq("F"),
        c_charge_desc_assault=df.c_charge_desc.str.contains("Assault"),
        c_charge_desc_battery=df.c_charge_desc.str.contains("Battery"),
        c_charge_desc_burglary=df.c_charge_desc.str.contains("Burg"),
        c_charge_desc_driving=df.c_charge_desc.str.contains("Driv"),
        c_charge_desc_dui=df.c_charge_desc.eq("Driving Under The Influence")
        | df.c_charge_desc.str.contains("DUI"),
        c_charge_desc_no_charge=df.c_charge_desc.eq("arrest case no charge"),
        c_charge_desc_possession=df.c_charge_desc.str.startswith("Pos"),
        c_charge_desc_theft=df.c_charge_desc.str.startswith("Theft"),
    ).drop(["c_charge_degree", "c_charge_desc"], axis=1)
    df = pd.get_dummies(df, columns=["age_cat", "race"])
    return df


class Compas(Dataset):
    """The COMPAS dataset. A standard benchmark for ML fairness algorithms.
    https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis
    """

    def __init__(self):
        # self.openml_id = 42193
        self.dataset = pd.DataFrame()
        self.cat_cols = [
            "sex",
            "race",
            "two_year_recid",
            "c_charge_degree_felony",
            "c_charge_desc_assault",
            "c_charge_desc_battery",
            "c_charge_desc_burglary",
            "c_charge_desc_driving",
            "c_charge_desc_dui",
            "c_charge_desc_no_charge",
            "c_charge_desc_possession",
            "c_charge_desc_theft",
            "race_African-American",
            "race_Asian",
            "race_Caucasian",
            "race_Hispanic",
            "race_Native American",
            "race_Other",
        ]
        self.path = os.path.expanduser(os.path.join(CACHE_PATH, "compas.parquet"))
        super().__init__()
        self.download()

    def cache_exists(self):
        os.path.isfile(self.path)

    def download(self):
        if self.cache_exists():
            self.dataset = pd.read_parquet(self.path)
        else:
            url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
            with urlopen(url) as f:
                csvdata = f.read().decode("utf-8")
                df = pd.read_csv(StringIO(csvdata))
                self.dataset = df[
                    [
                        "sex",
                        "age",
                        "age_cat",
                        "race",
                        "c_charge_degree",
                        "c_charge_desc",
                        "juv_fel_count",
                        "juv_misd_count",
                        "juv_other_count",
                        "priors_count",
                        "two_year_recid",
                    ]
                ]
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self.dataset.to_parquet(self.path)
        # self.dataset = download_openml(self.openml_id)[0]

    @property
    def columns(self):
        return self.dataset.columns

    def __getitem__(self, item):
        try:
            return self.dataset[item]
        except ValueError as exc:
            raise KeyError(item) from exc

    def get_xy(self, group_var: Optional[str] = None) -> Tuple[NDArray, NDArray]:
        df = process(self.dataset)
        x = df.drop("two_year_recid", axis=1).to_numpy().astype(float)
        if group_var:
            group = df[group_var]
            y = df[["two_year_recid"]].assign(**{group_var: group}).to_numpy().astype(float)
        else:
            y = df["two_year_recid"].to_numpy().astype(float)
        return x, y

    def cat_index(self, col):
        if col not in self.cat_cols:
            raise ValueError(f"Column {col} is not categorical.")
        return self.dataset[col].factorize(sort=True)[1]

    # def get_xy(self, group_var: Optional[str] = None):
    #     x = self.dataset.drop("class", axis=1)
    #     x = pd.get_dummies(x, columns=self.cat_cols).to_numpy().astype(float)
    #     if group_var:
    #         group = self.dataset[group_var].factorize(sort=True)[0]
    #         y = self.dataset[["class"]].assign(**{group_var: group}).to_numpy().astype(float)
    #     else:
    #         y = self.dataset["class"].to_numpy().astype(float)
    #     return x, y
