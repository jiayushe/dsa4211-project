import pandas as pd
import numpy as np
from sklearn import ensemble, metrics, model_selection, preprocessing
import matplotlib.pyplot as plt

# ================================================================== #
#                           preprocessing                            #
# ================================================================== #

## import data
df_train = pd.read_csv("train-xy.csv")
df_test = pd.read_csv("test-x.csv")

## drop column X55 since it has lots of missing values
df_train = df_train.drop(columns='X55')
df_test = df_test.drop(columns='X55')

## filter predictors based on pearson correlation coefficient with response
# df_corr = df_train.copy()
# df_corr = df_corr.corr(method="pearson").loc[["Y"]]
# df_corr = df_corr.abs()
# cols = []
# for col in df_corr.columns:
#     if df_corr[col]["Y"] > 0.05:
#         cols.append(col)
# cols = cols[1:]
cols = ['X7', 'X8', 'X9', 'X10', 'X18', 'X26', 'X27', 'X29', 'X30', 'X32', 'X33', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X51', 'X70', 'X71', 'X74', 'X75', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86', 'X90', 'X91', 'X93', 'X94', 'X95', 'X97', 'X98', 'X99']
df_train = df_train[cols+["Y"]]
df_test = df_test[cols]
df_test["Y"] = 0

## scale dataset
scaler_x = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
scaler_y = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
## train set
scaler_x.fit(df_train.drop("Y", axis=1))
train_x = scaler_x.transform(df_train.drop("Y", axis=1))
df_train_scaled = pd.DataFrame(train_x, columns=df_train.drop("Y", axis=1).columns, index=df_train.index)
df_train_scaled["Y"] = scaler_y.fit_transform(df_train["Y"].values.reshape(-1,1))
df_train = df_train_scaled
## test set
test_x = scaler_x.transform(df_test.drop("Y", axis=1))
df_test_scaled = pd.DataFrame(test_x, columns=df_test.drop("Y", axis=1).columns, index=df_test.index)
df_test = df_test_scaled

## feature selection
# model = ensemble.RandomForestRegressor(n_estimators=100, criterion="mse", random_state=0)
# X=df_train.drop("Y",axis=1).values
# y=df_train["Y"].values
# X_names=df_train.drop("Y",axis=1).columns.tolist()
# model.fit(X,y)
# impt = model.feature_importances_
# df_impt = pd.DataFrame({"IMPORTANCE":impt, "VARIABLE":X_names}).sort_values("IMPORTANCE", ascending=False)
# df_impt['cumsum'] = df_impt['IMPORTANCE'].cumsum(axis=0)
# df_impt = df_impt.set_index("VARIABLE")
# fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(15,5))
# fig.suptitle("Features Importance", fontsize=20)
# ax[0].title.set_text('variables')
# df_impt[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
# ax[0].set(ylabel="")
# ax[1].title.set_text('cumulative')
# df_impt[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
# ax[1].set(xlabel="", xticks=np.arange(len(df_impt)), xticklabels=df_impt.index)
# plt.xticks(rotation=70)
# plt.grid(axis='both')
# plt.show()
X_names = ["X7", "X8", "X9", "X10"]
X_train = df_train[X_names].values
y_train = df_train["Y"].values

# ================================================================== #
#                              training                              #
# ================================================================== #

## parameter tuning
param_dic = {
    'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
    'n_estimators':[100,250,500,750,1000,1250,1500,1750],
    'max_depth':[2,3,4,5,6,7],
    'min_samples_split':[2,4,6,8,10,20,40,60,100],
    'min_samples_leaf':[1,3,5,7,9],
    'max_features':[2,3,4,5,6],
    'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]
}
model_base = ensemble.GradientBoostingRegressor()
random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=1000, scoring="r2").fit(X_train, y_train)
model = random_search.best_estimator_
# grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring="r2").fit(X_train, y_train)
# model = grid_search.best_estimator_

## k-fold validation
# scores = []
# cv = model_selection.KFold(n_splits=5, shuffle=True)
# fig = plt.figure(figsize=(10,10))
# i = 1
# for train, test in cv.split(X_train, y_train):
#     prediction = model.fit(X_train[train], y_train[train]).predict(X_train[test])
#     truth = y_train[test]
#     score = metrics.r2_score(truth, prediction)
#     scores.append(score)
#     plt.scatter(prediction, truth, lw=2, alpha=0.3, label='Fold %d (R2 = %0.2f)' % (i,score))
#     i = i+1
# plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)], linestyle='--', lw=2, color='black')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('K-fold Validation')
# plt.legend()
# plt.show()

## save model
# from sklearn.externals import joblib
# joblib.dump(model, 'model.pkl')

## load model
# from sklearn.externals import joblib
# model = joblib.load('model.pkl')

# ================================================================== #
#                             prediction                             #
# ================================================================== #

## predition
X_test = df_test[X_names].values
predicted = model.predict(X_test)
predicted = scaler_y.inverse_transform(predicted.reshape(-1,1)).reshape(-1)

## output results as csv
res = pd.DataFrame(predicted)
res.to_csv("res.csv", header=["Y"], index=False)
