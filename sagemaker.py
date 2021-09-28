import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import tarfile
import sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()
region = sess.boto_region_name
bucket = sess.default_bucket()
print(region,bucket)

data = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=42)
train_data = pd.concat([X_train,y_train],axis=1)
train_data.to_csv("train_data.csv")
test_data = pd.concat([X_test,y_test],axis=1)
test_data.to_csv("test_data.csv")

train_path = sess.upload_data(path="train_data.csv",key_prefix="train_data")
test_path = sess.upload_data(path="test_data.csv",key_prefix="test_data")

from sagemaker.sklearn.estimator import SKLearn
sklearn_estimator = SKLearn(
    entry_point = "train_scripts.py",
    role = get_execution_role(),
    instance_count = 1,
    instance_type = "ml.m4.xlarge",
    framework_version = "0.23-1",
    base_job_name = "rt-scikit",
    metric_definitions = [{"Name":"accuracy","Regex":"accuracy of this model is (0[.][0-9]+)"}],
    hyperparameters = {
        "max_depth":3,
        "target":"target",
    }
)

sklearn_estimator.fit({"train":train_path,"test":test_path},wait=False)