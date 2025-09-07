import narwhals as nw
import numpy as np
from narwhals.typing import IntoFrame
from numpy import typing as npt

from .centroids import calc_centroids
from .graph import seek_valid_edges

from ._rblaidd import abc

def color_df(
    native_df: IntoFrame,
    x_col: str,
    y_col: str,
    label_col: str,
) -> npt.ArrayLike:
    df = nw.from_native(native_df)

    centroid_rows: list[list[float]] = list()
    labels = list()

    for label, sub_df in df.group_by(label_col):
        centroid_row = [sub_df[x_col].mean(), sub_df[y_col].mean()]
        centroid_rows.append(centroid_row)
        labels.append(label[0])

    labels, centroids = calc_centroids(
        df=native_df,
        x_col=x_col,
        y_col=y_col,
        label_col=label_col
    )

    edges = seek_valid_edges(centroids)
    
    centroid_colors = abc(len(centroids), edges, 50, 10000)

    centroid_index_by_val = {
        label:i
        for i, label in enumerate(labels)
    }

    return np.array([centroid_colors[ centroid_index_by_val[label] ] for label in df[label_col] ])

