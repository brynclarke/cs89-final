import pandas as pd
import numpy as np

from transformers import PortoSeguroTransformer

def preprocess():
    train = pd.read_csv("kaggle-data/train.csv")
    test = pd.read_csv("kaggle-data/test.csv")

    feature_types = []
    for s in train.columns:
        if not s in ("id", "target") and not s.startswith("ps_calc_"):
            if s.endswith("_cat"):
                ftype = "cat"
            elif s.endswith("_bin"):
                ftype = "bin"
            else:
                ftype = "num"
            
            feature_types += [{
                "name": s,
                "type": ftype
            }]

    pst = PortoSeguroTransformer(feature_types)

    X_train = pst.fit_transform(train)
    y_train = train.target.values
    X_test = pst.transform(test)

    np.save("data/X_train.npy", X_train.astype("float32"))
    np.save("data/y_train.npy", y_train.astype("int32"))
    np.save("data/X_test.npy", X_test.astype("float32"))