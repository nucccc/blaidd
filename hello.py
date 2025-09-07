








""""""
from random import shuffle

import blaidd

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

def load_iris() -> pl.DataFrame:
    df = pl.read_csv('iris.csv')

    #creating an index col on the species
    species_dict = {
        label : i
        for i, label in enumerate(set(df['species']))
    }
    df = df.with_columns(
        pl.Series(
            name="species_index",
            values=[species_dict[label] for label in df['species']]
        )
    )

    return df


def build_reticulate(
    nx: int,
    ny: int,
    ret_size: int = 100,
    ret_var: float = 0.2,
) -> pl.DataFrame:
    cats = [i for i in range(nx * ny)]
    shuffle(cats)

    xs: list[np.array] = list()
    ys: list[np.array] = list()
    cs: list[np.array] = list()
    for i in range(ny):
        for j in range(nx):
            xs.append(np.random.randn(ret_size) * ret_var + i)
            ys.append(np.random.randn(ret_size) * ret_var + j)
            cs.append(np.ones(ret_size, dtype=np.uint8) * cats[(i*nx + j)])
    x = np.concat(xs)
    y = np.concat(ys)
    c = np.concat(cs)

    return pl.DataFrame({
        'x':x,
        'y':y,
        'c':c,
    })


if __name__ == '__main__':
    df = load_iris()

    """plt.scatter(
        x = df['sepal_length'],
        y = df['sepal_width'],
        c = df['species_index'],
    )
    plt.show()"""

    df = build_reticulate(8, 12, ret_size=1000, ret_var=0.3)

    """plt.scatter(
        x = df['x'],
        y = df['y'],
        c = df['c'],
        s = 4
    )
    plt.show()"""

    df = df.with_columns(
        color_graphed=blaidd.color_df(
            df,
            'x',
            'y',
            'c'
        )
    )

    plt.scatter(
        x = df['x'],
        y = df['y'],
        c = df['c'],
        s = 4
    )
    plt.show()

    plt.scatter(
        x = df['x'],
        y = df['y'],
        c = df['color_graphed'],
        s = 4
    )
    plt.show()
