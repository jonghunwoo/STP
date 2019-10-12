# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import xgboost
from vecstack import stacking

import warnings
warnings.filterwarnings('ignore')

########################################### MakingLT ###################################################################
# Data 불러오기
ST_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del ST_PaintingLT['Unnamed: 0']

# Data Type 변경
ST_PaintingLT["Emergency"] = ST_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding
ST_p_Emergency_one_hot_encoded = pd.get_dummies(ST_PaintingLT.Emergency)
ST_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(ST_PaintingLT.ApplyLeadTime)
ST_p_STG_one_hot_encoded = pd.get_dummies(ST_PaintingLT.STG)
ST_p_Service_one_hot_encoded = pd.get_dummies(ST_PaintingLT.Service)
ST_p_Pass_one_hot_encoded = pd.get_dummies(ST_PaintingLT.Pass)
ST_p_Sch_one_hot_encoded = pd.get_dummies(ST_PaintingLT.Sch)
ST_p_Material_one_hot_encoded = pd.get_dummies(ST_PaintingLT.Material)
ST_p_Making_Co_one_hot_encoded = pd.get_dummies(ST_PaintingLT.Making_Co)
ST_p_After2_Co_one_hot_encoded = pd.get_dummies(ST_PaintingLT.After2_Co)

# Input, Output 분류
ST_p_Inputdata = pd.concat((ST_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , ST_p_Emergency_one_hot_encoded
                      , ST_p_ApplyLeadTime_one_hot_encoded
                      , ST_p_STG_one_hot_encoded
                      , ST_p_Service_one_hot_encoded
                      , ST_p_Pass_one_hot_encoded
                      , ST_p_Sch_one_hot_encoded
                      , ST_p_Material_one_hot_encoded
                      , ST_p_Making_Co_one_hot_encoded
                      , ST_p_After2_Co_one_hot_encoded)
                      , axis=1)

ST_p_Outputdata = ST_PaintingLT[['PaintingLT']]

# Vector로 변환 ; Deeplearning에서는 Vector 사용
ST_X1 = ST_p_Inputdata.values
ST_Y1 = ST_p_Outputdata.values

# Training Data, Test Data 분리
ST_X1_train, ST_X1_test, ST_Y1_train, ST_Y1_test = train_test_split(ST_X1, ST_Y1, test_size = 0.33, random_state = 42)

# Stacking 모델 구축 (decisiontree + randomforest +xgboost - regression)
painting_stacking_model1 = [
    DecisionTreeRegressor(max_depth=10, random_state=42),
    RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42),
    xgboost.XGBRegressor(max_depth=10, learning_rate=0.5, n_estimators=100, seed=0)]

ST1_train, ST1_test = stacking(painting_stacking_model1, ST_X1_train, ST_Y1_train, ST_X1_test,
                               regression=True, n_folds=4, stratified=True, shuffle=True, random_state=42, verbose=2)

painting_stacking_model2 = xgboost.XGBRegressor(max_depth=20, colsample_bytree=0.5, learning_rate=0.5, n_estimators=100, seed=0)

painting_stacking_model2.fit(ST1_train, ST_Y1_train)

ST1_p_predicted = painting_stacking_model2.predict(ST1_test)
ST1_p_predicted[ST1_p_predicted<0] = 0

ST1_length_x1test = len(ST1_test)
ST1_p_predicted = ST1_p_predicted.reshape(ST1_length_x1test,1)

ST1_p_mae = abs(ST1_p_predicted - ST_Y1_test).mean(axis=0)
ST1_p_mape = (np.abs((ST1_p_predicted - ST_Y1_test) / ST_Y1_test).mean(axis=0))
ST1_p_rmse = np.sqrt(((ST1_p_predicted - ST_Y1_test) ** 2).mean(axis=0))
ST1_p_rmsle = np.sqrt((((np.log(ST1_p_predicted + 1) - np.log(ST_Y1_test + 1)) ** 2).mean(axis=0)))

print(ST1_p_mae)
print(ST1_p_mape)
print(ST1_p_rmse)
print(ST1_p_rmsle)

# Stacking 모델 구축 (decisiontree + randomforest +xgboost - classifier)
painting_stacking_model3 = [
    DecisionTreeClassifier(max_depth=20, random_state=42),
    RandomForestClassifier(max_depth=20, n_estimators=100, random_state=42),
    xgboost.XGBClassifier(max_depth=10, learning_rate=0.5, n_estimators=100, seed=0)]

ST2_train, ST2_test = stacking(painting_stacking_model3, ST_X1_train, ST_Y1_train, ST_X1_test,
                               regression=True, n_folds=4, stratified=True, shuffle=True, random_state=42, verbose=2)

painting_stacking_model4 = xgboost.XGBClassifier(max_depth=20, colsample_bytree=0.5, learning_rate=0.5, n_estimators=100, seed=0)

painting_stacking_model4.fit(ST2_train, ST_Y1_train)

ST2_p_predicted = painting_stacking_model4.predict(ST2_test)

ST2_length_x1test = len(ST2_test)
ST2_p_predicted = ST2_p_predicted.reshape(ST2_length_x1test,1)

ST2_p_mae = abs(ST2_p_predicted - ST_Y1_test).mean(axis=0)
ST2_p_mape = (np.abs((ST2_p_predicted - ST_Y1_test) / ST_Y1_test).mean(axis=0))
ST2_p_rmse = np.sqrt(((ST2_p_predicted - ST_Y1_test) ** 2).mean(axis=0))
ST2_p_rmsle = np.sqrt((((np.log(ST2_p_predicted + 1) - np.log(ST_Y1_test + 1)) ** 2).mean(axis=0)))

print(ST2_p_mae)
print(ST2_p_mape)
print(ST2_p_rmse)
print(ST2_p_rmsle)

########################################################################################################################
# 분석결과 저장
ST_p_evaluation = {'MAE' :  [ST1_p_mae[0], ST2_p_mae[0]], 'MAPE' :  [ST1_p_mape[0], ST2_p_mape[0]], 'RMSE' :  [ST1_p_rmse[0], ST2_p_rmse[0]], 'RMSLE' : [ST1_p_rmsle[0], ST2_p_rmsle[0]]}

ST_p_evaluation = pd.DataFrame(ST_p_evaluation, index = ['painting_stacking_regression','painting_stacking_classifier'])

print(ST_p_evaluation)

# 분석결과 .csv 파일 저장
ST_p_evaluation.to_csv('spool_p_stacking_conclusion.csv', sep=',', na_rep='NaN')