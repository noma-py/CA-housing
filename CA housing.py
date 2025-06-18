import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,PredictionErrorDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


#What do i want to predict?
#I want to predict housing prices(regression)

data=pd.read_csv("/Users/maxnordstrom/python/SCIKIT-LEARN/CA housing/housing.csv")

USE_GRIDSEARCH=False
USE_RANDOMSEARCH=False

X=data.drop(columns=["median_house_value"])
y=np.log(data["median_house_value"])

#add engineered features to improve model performance
X["rooms-per_household"]=X["total_rooms"]/X["households"]
X["bedrooms_per_room"]=X["total_bedrooms"]/X["total_rooms"]
X["population_per_household"]=X["population"]/X["households"]

#split featured data into numerical and categorical columns
numeric=X.select_dtypes(include=["int64","float64"]).columns.to_list()
categoric=X.select_dtypes(exclude=["int64","float64"]).columns.to_list()

#build preprocessing model for numerical data (handle missing values adn scale)
num_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

#build preprocessing model for categorical data (impute and one-hot encode)
cat_pipeline=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="constant",fill_value="missing")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

#combine numerical and categorical pipelines into a single ColumnTransformer
preprocessor=ColumnTransformer(
    transformers=[
        ("num",num_pipeline,numeric),
        ("cat",cat_pipeline,categoric)
    ])

#split into training and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#create full pipeline with preprocessing and model 
pipe=Pipeline([
    ("preprocessor",preprocessor),
    ("model",RandomForestRegressor(n_estimators=250,min_samples_split=2,max_depth=20))
])

#optional: search for best hyperparameters using GridSearchCV or RandomizedSearchCV

param_grid={
    "model__n_estimators":[150,250],
    "model__max_depth":[None,10,20],
    "model__min_samples_split":[2,5]
}

if USE_GRIDSEARCH:
    search=GridSearchCV(pipe,param_grid,cv=5,n_jobs=-1)
    search.fit(X_train,y_train)
elif USE_RANDOMSEARCH:
    search=RandomizedSearchCV(pipe,param_grid, n_iter=20, cv=5,scoring="r2",n_jobs=-1)
    search.fit(X_train,y_train)
else:
    search=pipe



#train model
search.fit(X_train,y_train) 

#make predictions on test data 
y_pred=search.predict(X_test)

if USE_GRIDSEARCH or USE_RANDOMSEARCH:
    print("Best r2 score: ",search.best_score_)
    print("best parameter: ",search.best_params_)

print("R2 score: ",r2_score(y_test,y_pred))


display=PredictionErrorDisplay(y_true=y_test,y_pred=y_pred)
display.plot()
plt.show()

residuals=y_test-y_pred
plt.scatter(y_pred,residuals)
plt.axhline(0,color="black",linestyle="--")
plt.xlabel("predicted")
plt.ylabel("residual")
plt.title("residual plot")
plt.show()

plt.hist(residuals,bins=50)
plt.xlabel("residuals")
plt.title("residual distribution")
plt.show()