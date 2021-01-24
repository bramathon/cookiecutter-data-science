import argparse
import yaml
import logging

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score

## Logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--params", "-p", default="params.yaml", type=str, required=True, help="Settings file for the training")
    parser.add_argument("--out_file", "-o", default="models/preds.csv", type=str, help="Output file")
    parser.add_argument("--log", "-l", default="info", type=str, help="Set the logging level")
    args = parser.parse_args()

    ## Logging Setup
    loglevel = args.log
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logging.basicConfig(
        level=numeric_level, format="[{asctime}] {levelname:<4.5} - {name:<20.20} | {message}", datefmt="%H:%M:%S", style="{"
    )

    ## Load settings
    with open(args.params, "r") as f:
        logger.info(f"Using settings file: {args.params}")
        params = yaml.load(f, Loader=yaml.Loader)

    priors = np.array(params['priors'])
    # ensure the priors sum to 1.0
    priors = priors/priors.sum()
    
    ## Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB(priors=priors)

    ## Train
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    ## Evaluate
    acc = balanced_accuracy_score(y_test,y_pred)
    kpis = dict(balanced_accuracy=float(acc)) # make sure to convert from numpy to python type
    print("Number of mislabeled points out of a total {} points : {}".format(X_test.shape[0], (y_test != y_pred).sum()))

    # Save
    np.savetxt(args.out_file, y_pred, delimiter=',')
    with open("models/metrics.yaml", "w") as f:
        yaml.dump(kpis, f)
        
