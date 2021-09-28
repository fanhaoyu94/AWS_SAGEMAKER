import argparse
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir,"model.joblib"))

def process_args():
    '''
    parse the arguments pass into
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimator",type=int,default=50)
    parser.add_argument("--max_depth",type=int,default=5)
    parser.add_argument("--model-dir",type=str,default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train",type=str,default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test",type=str,default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--target",type=str,default="target")

    return parser.parse_args()

def train(args):
    train_file_path = os.path.join(args.train,os.listdir(args.train)[0])
    print(args.train,train_file_path)
    test_file_path = os.path.join(args.test,os.listdir(args.test)[0])
    train_df = pd.read_csv(train_file_path)
    feature_columns = [column for column in train_df.columns if column != args.target]
    train_feature, train_target = train_df[feature_columns],train_df['target']
    test_df = pd.read_csv(test_file_path)
    test_feature, test_target = test_df[feature_columns],test_df['target']
    
    print("Model is Training")
    RF = RandomForestClassifier(
        n_estimators=args.n_estimator,
        max_depth=args.max_depth,
        n_jobs = -1
        )
    RF_model = RF.fit(train_feature,train_target)
    print("Validating Models")
    accuracy = round(accuracy_score(test_target,RF_model.predict(test_feature)),2)
    print(f"accuracy of this model is {accuracy}")

    #save models
    path = os.path.join(args.model_dir,"model.joblib")
    joblib.dump(RF_model,path)

if __name__ == "__main__":
    args = process_args()
    train(args)


