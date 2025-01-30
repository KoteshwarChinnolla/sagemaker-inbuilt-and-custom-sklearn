import argparse
import joblib
import os
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# inference functions ---------------


if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--C', type=int, default=os.environ.get("SM_HP_C"))
    parser.add_argument('--gamma', type=int, default=os.environ.get("SM_HP_GAMMA"))

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args, _ = parser.parse_known_args()

    print("-"*100)
    print("reading data")

    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    data = pd.concat(raw_data)
    index_names = data[ data[0] == 'product' ].index 
    data.drop(index_names,inplace=True)
    x = data.iloc[:,[0,1,3]].values
    y = data.iloc[:, [6]].values
    print("-"*100)
    print(data.head(5))
    print("building training and testing datasets")
    X_train,X_test,y_train,y_test= train_test_split(x, y, test_size = 0.175, random_state=0)

    #encoding data

    ohe = OneHotEncoder()
    ohe.fit(data[[0]])
    ct=make_column_transformer((OneHotEncoder(),[0]),remainder='passthrough')

    # train
    print("training model")

    svm=SVC(C=args.C, gamma=args.gamma)
    model=make_pipeline(ct,svm)
    model.fit(X_train, y_train)

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        print("input data loaded", df.head())

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    print("Serializing the generated output.", prediction[0])
    return prediction[0]


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    print("input data", input_data)
    features = model.predict(input_data)
    print("prediction done", features)
    return features

def model_fn(model_dir):
    print(model_dir)
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf