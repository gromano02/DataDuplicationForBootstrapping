import numpy as np
import pandas as pd
import argparse
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from ucimlrepo import fetch_ucirepo


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

warnings.filterwarnings('ignore')

def duplicate_data(X, y, factor):
    X_dup = np.tile(X, (factor, 1))
    y_dup = np.tile(y, factor)
    return X_dup, y_dup

def calculate_tree_correlations(model, X, y):

    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
   
    num_trees = tree_predictions.shape[0]

    tree_fscores = [f1_score(y, tree_predictions[i], zero_division=1) for i in range(num_trees)]
    avg_fscore = np.mean(tree_fscores) if tree_fscores else 0
    
    std_devs = np.std(tree_predictions, axis=1, ddof=1)
    denominator = np.mean([std_devs[i] * std_devs[j] for i in range(num_trees) for j in range(num_trees)])
    
    tree_margins = np.zeros((len(y), num_trees))

    # Compute raw margin function for each tree
    for i, (tree, preds) in enumerate(zip(model.estimators_, tree_predictions)):
        # Compute raw margin as I(h(x,T) = y) - I(h(x,T) = j(x,y))
        correct_preds = (preds == y).astype(int)  # 1 if correct, 0 otherwise
        # Identify the "best" incorrect class
        proba = tree.predict_proba(X)  # Get class probabilities
        proba[np.arange(len(y)), y.astype(int)] = -np.inf
        best_wrong_class = np.argmax(proba, axis=1)  # Get index of most confident wrong class
        
        incorrect_preds = (preds == best_wrong_class).astype(int)  # 1 if classified as best wrong class
        tree_margins[:, i] = correct_preds - incorrect_preds  # Compute raw margin
    
            
    correlation_matrix = np.corrcoef(tree_margins, rowvar=False)
    
    numerator = np.mean([correlation_matrix[i, j] * std_devs[i] * std_devs[j] 
                    for i in range(num_trees) for j in range(num_trees)])
    mean_correlation = numerator / denominator if denominator != 0 else 0
    
    margin_function = np.mean(tree_margins, axis=1)
    
    # Compute strength as expectation over (X, Y)
    strength = np.mean(margin_function)
    
    return mean_correlation, strength, avg_fscore

data = pd.read_csv("mammography.csv", header=None)
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].astype(str).str.strip("'\" ").astype(int).values  # Labels as integers
y = np.where(y == -1, 0, y)


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a Random Forest model with different methods and duplication factors.")
parser.add_argument("--method", type=str, choices=['standard', 'class_weighted', 'undersampling'], required=True, 
                    help="The method to use: 'standard', 'class_weighted', or 'undersampling'.")
parser.add_argument("--factor", type=int, required=True, help="Duplication factor for training data.")

args = parser.parse_args()
method = args.method
factor = args.factor


hyperparams = [[28.73205985, 6.47301361, 17.84560384, 15.10946848],
 [15.49165039, 49.48588494, 10.98595875, 7.12403051],
 [636.9177093, 35.76806537, 7.25297576, 17.2597332 ],
 [380.68799145, 43.08701634, 5.25160169, 10.35464707],
 [530.2284922, 30.75474413, 8.32991538, 10.54921443],
 [22.98071007, 18.05653512, 19.56534573, 14.20005301],
 [246.13714858, 2.34079641, 15.67625774, 6.56012679],
 [83.49903301, 14.9818116, 12.63775737, 18.6959126 ],
 [48.23761066, 28.76411886, 13.11649229, 2.93714757],
 [136.75716838, 20.89601936, 2.23684806, 2.32996772]]


outer_fscore_results = []
outer_precision_results = []
outer_recall_results = []


kf_outer = StratifiedKFold(n_splits=5, shuffle=True)
param_scores = {tuple(params): [] for params in hyperparams}  # To store scores for each hyperparameter set

for train_index, test_index in kf_outer.split(X, y):
    
  tree_correlations = []
  tree_fscores = []
  tree_strengths = []

  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  kf_inner = StratifiedKFold(n_splits=5, shuffle=True)

  for train_val_index, val_index in kf_inner.split(X_train, y_train):
      X_train_inner, X_val = X_train[train_val_index], X_train[val_index]
      y_train_inner, y_val = y_train[train_val_index], y_train[val_index]
      X_train_inner, y_train_inner = duplicate_data(X_train_inner, y_train_inner, factor)
      for params in hyperparams:
        n_estimators, max_depth, min_samples_split, min_samples_leaf = map(int, params)
        if method == 'standard':
          model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf)
        elif method == 'class_weighted':
          model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           class_weight='balanced')
        elif method == 'undersampling':
          model = BalancedRandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf)
        model.fit(X_train_inner, y_train_inner)
        y_pred_val = model.predict(X_val)
        
        fscore = f1_score(y_val, y_pred_val, zero_division=1)
        param_scores[tuple(params)].append(fscore)

  avg_scores = {params: np.mean(scores) for params, scores in param_scores.items()}
  best_inner_params = max(avg_scores, key=avg_scores.get)

  n_estimators, max_depth, min_samples_split, min_samples_leaf = map(int, best_inner_params)
  if method == 'standard':
    best_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
  elif method == 'class_weighted':
    best_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced'
            )
  elif method == 'undersampling':
    best_model = BalancedRandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
  X_train, y_train = duplicate_data(X_train, y_train, factor)
  best_model.fit(X_train, y_train)
  y_pred_test = best_model.predict(X_test)

        # Evaluate on the outer test set
  precision = precision_score(y_test, y_pred_test, zero_division=1)
  recall = recall_score(y_test, y_pred_test, zero_division=1)
  fscore = f1_score(y_test, y_pred_test, zero_division=1)
  outer_fscore_results.append(fscore)
  outer_precision_results.append(precision)
  outer_recall_results.append(recall)
  
  

      # Average performance across outer folds
avg_outer_fscore = np.mean(outer_fscore_results)
avg_outer_precision = np.mean(outer_precision_results)
avg_outer_recall = np.mean(outer_recall_results)


# Get the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
# Define the "data" directory inside the script's location
data_dir = os.path.join(base_dir, "Data")
# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

filename = os.path.join(data_dir, f"mammography_{method}.csv")

write_header = not os.path.exists(filename)
with open(filename, 'a') as f:
  if write_header:
    f.write("Factor, Method, F-score, Precision, Recall\n")  # Write the header once
  f.write(f"{factor},{method},{avg_outer_fscore:.4f},{avg_outer_precision:.4f},{avg_outer_recall:.4f}\n")
  
if isinstance(best_model, (RandomForestClassifier, BalancedRandomForestClassifier)):
    tree_corr, tree_strength, tree_fscore = calculate_tree_correlations(best_model, X_test, y_test)
    
    filename2 = os.path.join(data_dir, f"mammography_decision_trees_{method}.csv")
    
    tree_write_header = not os.path.exists(filename2)

    with open(filename2, 'a') as f:
        if tree_write_header:
            f.write("Factor,Method,F1-Score,Correlation,Strength\n")
        f.write(f"{factor},{method},{tree_fscore:.4f},{tree_corr:.4f},{tree_strength:.4f}\n")





