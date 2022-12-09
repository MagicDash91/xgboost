import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

st.header('Train your own XGBoost model')

df = pd.read_csv('diabetes.csv')
st.dataframe(df)

diabetes_df_copy = df.copy(deep = True) #deep = True -> Buat salinan indeks dan data dalam dataframe
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Showing the Count of NANs
print(diabetes_df_copy.isnull().sum())

diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].median(), inplace = True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].median(), inplace = True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace = True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace = True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace = True)

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = diabetes_df_copy[(diabetes_df_copy['Outcome']==0)] # semua data yang value outcome nya = 0
df_minority = diabetes_df_copy[(diabetes_df_copy['Outcome']==1)] # semua data yang value outcome nya = 1
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 n_samples= 500, # to match majority class, menyamakan jumlah value 1 dengan 0
                                 random_state=0)  # reproducible results, random state 0 is better than 42
                                                  #Random state = Mengontrol pengacakan yang diterapkan ke data agar hasil yang didapatkan tetap sama
# Combine majority class with upsampled minority class
diabetes_df_copy2 = pd.concat([df_minority_upsampled, df_majority])

import scipy.stats as stats
z = np.abs(stats.zscore(diabetes_df_copy2))
data_clean = diabetes_df_copy2[(z<3).all(axis = 1)] 

X = data_clean.drop('Outcome', axis=1)
y = data_clean['Outcome']

split = st.sidebar.slider('Choose the test size', 1, 99, 10)
splittrain = 100 - split
split2 = split/100
rd = st.sidebar.slider('Choose the train test split random state', 0, 42, 0)

st.sidebar.write("**XGBoost Parameters**")

gamma = st.sidebar.slider('Choose the XGBoost gamma', 0, 1, 0)
max_d = st.sidebar.slider('Choose the XGBoost Maximum Depth', 4, 10, 6)

st.write("Your train size : ", splittrain)
st.write("Your test size : ", split)
st.write("Your train test split random state : ", rd)
st.write("Your XGBoost Gamma : ", gamma)
st.write("Your XGBoost Maximum Depth : ", max_d)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split2, random_state=rd)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

xgb=XGBClassifier(max_depth=max_d, gamma=gamma)
xgb.fit(X_train,y_train)
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train,y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
y_pred = xgb.predict(X_test)
acc = round(accuracy_score(y_test, y_pred)*100 ,2)
y_pred = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
figure = plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


f12 = round(f1*100,2)
prec2 = round(prec*100,2)
recall2 = round(recall*100,2)

from xgboost import plot_importance
from matplotlib import pyplot
feature_importance = plot_importance(model)
plt.figure(figsize=(30,45))
pyplot.show()


st.write("**Algorithm Accuracy in (%)**")
st.info(acc)
st.write("**Precision**")
st.info(prec2)
st.write("**Recall**")
st.info(recall2)
st.write("**F-1 Score**")
st.info(f12)
st.write("**Confusion Matrix**")
st.write(figure)




# fit model on training data
model = XGBClassifier()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# make predictions for test data
predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.xlabel('Epoch')
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()

st.pyplot(fig)

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.xlabel('Epoch')
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

st.pyplot(fig)
