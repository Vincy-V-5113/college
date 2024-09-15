import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
collegeData = pd.read_csv("/content/Admission_Predict_Ver1.1.csv")
trimColNames = [name.strip() for name in collegeData.columns]
collegeData.columns = trimColNames

collegeData.head()

collegeData = collegeData.drop("Serial No.", axis = 1)

collegeData["Research"].dtype

collegeData["Research"] = collegeData["Research"].astype('category')

correlation_matrix = collegeData.iloc[:,:].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
sns.catplot(data = collegeData, x = "Research", y = "Chance of Admit")

X = collegeData.iloc[:,0:7]
y = collegeData.iloc[:,7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1005)

rf = RandomForestRegressor(n_estimators=250,
                           max_features=(2/7),
                           min_samples_split=5,
                           n_jobs=2,
                           random_state=1005)

rf.fit(X_train, y_train)

train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)

train_mse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_mse = np.sqrt(mean_squared_error(y_test, test_predictions))
train_r2 = rf.score(X_train, y_train)
test_r2 = rf.score(X_test, y_test)

print("Train MSE ::", train_mse)
print("Test MSE ::", test_mse)
print("Train R^2 ::", train_r2)
print("Test R^2 ::", test_r2)

features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

newPerson = [[330, 110, 4, 4.5, 4.5, 9.5, 0]]
pred = rf.predict(newPerson)
pred[0]
!pip install streamlit

%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
collegeData = pd.read_csv("/content/Admission_Predict_Ver1.1.csv")
trimColNames = [name.strip() for name in collegeData.columns]
collegeData.columns = trimColNames

# Drop the 'Serial No.' column
collegeData = collegeData.drop("Serial No.", axis=1)

# Convert 'Research' column to categorical
collegeData["Research"] = collegeData["Research"].astype('category')

# Split features and target variable
X = collegeData.iloc[:, 0:7]
y = collegeData.iloc[:, 7]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1005)

# Train the Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=250,
                           max_features=(2/7),
                           min_samples_split=5,
                           n_jobs=2,
                           random_state=1005)
rf.fit(X_train, y_train)

# Predictions
train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)

# Calculate MSE and R^2
train_mse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_mse = np.sqrt(mean_squared_error(y_test, test_predictions))
train_r2 = rf.score(X_train, y_train)
test_r2 = rf.score(X_test, y_test)

# Feature importances
features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

# Streamlit app
st.title('College Admission Prediction')

# Display dataset
st.write("### Dataset")
st.write(collegeData.head())

# Display correlation heatmap
st.write("### Correlation Heatmap")
correlation_matrix = collegeData.iloc[:, :].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
st.pyplot()

# Display catplot
st.write("### Catplot")
sns.catplot(data=collegeData, x="Research", y="Chance of Admit")
st.pyplot()

# Display feature importances
st.write("### Feature Importances")
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
st.pyplot()

# Prediction for a new person
st.write("### Predict Admission Chance for a New Person")
new_person_input = st.text_input("Enter features for a new person (GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research)")
if st.button("Predict"):
    new_person_features = [float(x.strip()) for x in new_person_input.split(',')]
    pred = rf.predict([new_person_features])
    st.write(f"Predicted Admission Chance: {pred[0]}")

!npm install localtunnel
!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
