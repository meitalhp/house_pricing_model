import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold


#-----------------------------------------------------------------------------------------------------------------------------------------------
#Building the Pipeline:
from madlan_data_prep import prepare_data
smaller_data=prepare_data()
X = smaller_data.drop("price", axis=1)
y = smaller_data.price.astype(float)

from sklearn.model_selection import train_test_split
#stratify not working as we have a lot of single classes valeus
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#deviding the datat to numeric and categorial features:
num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O' and (X_train[col].dtypes!='category')]
cat_cols = [col for col in X_train.columns if (X_train[col].dtypes=='O' or (X_train[col].dtypes=='category'))]

numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False)),
    ('scaling', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

column_transformer = ColumnTransformer([
     ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)
    ], remainder='drop')


# Create the Elastic Net model
np.random.seed(2)
elastic_net_model = ElasticNet(alpha=0.33,  l1_ratio=0.99)

# Update the pipeline without feature selection
elastic_net_pipeline = Pipeline([
    ('preprocessing', column_transformer),
    ('elastic_net', elastic_net_model)
])

# Fit the pipeline
elastic_net_pipeline.fit(X_train, y_train)


# Calculate RMSE using cross-validation
cross_val = KFold(n_splits=10)
rmse_scores = np.sqrt(-cross_val_score(elastic_net_pipeline, X_train, y_train, scoring='neg_mean_squared_error', cv=cross_val))
mean_rmse = np.mean(rmse_scores)

# Calculate R-squared using cross-validation
r2_scores = cross_val_score(elastic_net_pipeline, X_train, y_train, scoring='r2', cv=cross_val)
mean_r2 = np.mean(r2_scores)

print("R-squared: ", mean_r2,"\nRMSE: ", mean_rmse)

#making predictions:
ypred= elastic_net_pipeline.predict(X_test)



from sklearn.metrics import r2_score,mean_squared_error as mse

MSE= mse(y_test,ypred)
RMSE= np.sqrt(MSE)
R2=r2_score(y_test, ypred)
print("test_MSE: ",MSE,'\ntest_RMSE: ',RMSE,'\ntest_R2: ',R2)
#----------------------------------------------------------------------------
#creating the pkl file:
import pickle
pickle.dump(elastic_net_pipeline, open("house_pricing_model.pkl","wb"))

