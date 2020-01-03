# 해당 파일은 앙상블학습의 stacking 알고리즘(다양한 학습 알고리즘의 조합)을 적용한 학습모델을 구축하여 성능 확인을 수행
# 테스트 후 가장 성능이 좋은 모델의 조합 2가지만 정리
# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from vecstack import stacking
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

########################################### MakingLT ###################################################################
# Data 불러오기
ST_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del ST_MakingLT['Unnamed: 0']

# Data Type 변경
ST_MakingLT["Emergency"] = ST_MakingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
ST_m_Emergency_one_hot_encoded = pd.get_dummies(ST_MakingLT.Emergency)
ST_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(ST_MakingLT.ApplyLeadTime)
ST_m_STG_one_hot_encoded = pd.get_dummies(ST_MakingLT.STG)
ST_m_Service_one_hot_encoded = pd.get_dummies(ST_MakingLT.Service)
ST_m_Pass_one_hot_encoded = pd.get_dummies(ST_MakingLT.Pass)
ST_m_Sch_one_hot_encoded = pd.get_dummies(ST_MakingLT.Sch)
ST_m_Material_one_hot_encoded = pd.get_dummies(ST_MakingLT.Material)
ST_m_Making_Co_one_hot_encoded = pd.get_dummies(ST_MakingLT.Making_Co)

# Input, Output 분류
ST_m_Inputdata = pd.concat((ST_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , ST_m_Emergency_one_hot_encoded
                      , ST_m_ApplyLeadTime_one_hot_encoded
                      , ST_m_STG_one_hot_encoded
                      , ST_m_Service_one_hot_encoded
                      , ST_m_Pass_one_hot_encoded
                      , ST_m_Sch_one_hot_encoded
                      , ST_m_Material_one_hot_encoded
                      , ST_m_Making_Co_one_hot_encoded)
                      , axis=1)

ST_m_Outputdata = ST_MakingLT[['MakingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
ST_X1 = ST_m_Inputdata.values
ST_Y1 = ST_m_Outputdata.values

# Training Data, Test Data 분리
ST_X1_train, ST_X1_test, ST_Y1_train, ST_Y1_test = train_test_split(ST_X1, ST_Y1, test_size = 0.33, random_state = 42)

##################################### meta model - xg boost ############################################################
# Stacking 모델 구축 (decisiontree + randomforest +xgboost - classifier)
making_stacking_model3 = [
    DecisionTreeClassifier(max_depth=20, random_state=42),
    RandomForestClassifier(max_depth=20, n_estimators=100, random_state=42),
    xgboost.XGBClassifier(max_depth=10, n_estimators=100, seed=0, learning_rate=0.5)]

ST2_train, ST2_test = stacking(making_stacking_model3, ST_X1_train, ST_Y1_train, ST_X1_test,
                               regression=True, n_folds=4, stratified=True, shuffle=True, random_state=42, verbose=2)

making_stacking_model4 = xgboost.XGBClassifier(max_depth=20, colsample_bytree=0.5, learning_rate=0.5, n_estimators=100, seed=0)

making_stacking_model4.fit(ST2_train, ST_Y1_train)

ST2_m_predicted = making_stacking_model4.predict(ST2_test)
ST2_m_predicted[ST2_m_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
ST2_length_x1test = len(ST2_test)
ST2_m_predicted = ST2_m_predicted.reshape(ST2_length_x1test,1)

# 학습 모델 성능 확인
ST2_m_mae = abs(ST2_m_predicted - ST_Y1_test).mean(axis=0)
ST2_m_mape = (np.abs((ST2_m_predicted - ST_Y1_test) / ST_Y1_test).mean(axis=0))
ST2_m_rmse = np.sqrt(((ST2_m_predicted - ST_Y1_test) ** 2).mean(axis=0))
ST2_m_rmsle = np.sqrt((((np.log(ST2_m_predicted + 1) - np.log(ST_Y1_test + 1)) ** 2).mean(axis=0)))

print(ST2_m_mae)
print(ST2_m_mape)
print(ST2_m_rmse)
print(ST2_m_rmsle)

########################################################################################################################
# Stacking 모델 구축 (randomforest + gradient boost + xgboost - classifier)
making_stacking_model11 = [
    RandomForestClassifier(max_depth=20, n_estimators=100, random_state=42),
    GradientBoostingClassifier(max_depth=20, n_estimators=100, random_state=42),
    xgboost.XGBClassifier(max_depth=10, n_estimators=100, seed=0)]

ST6_train, ST6_test = stacking(making_stacking_model11, ST_X1_train, ST_Y1_train, ST_X1_test,
                               regression=True, n_folds=4, stratified=True, shuffle=True, random_state=42, verbose=2)

making_stacking_model12 = xgboost.XGBClassifier(max_depth=10, colsample_bytree=0.5, learning_rate=0.5, n_estimators=100, seed=0)

making_stacking_model12.fit(ST6_train, ST_Y1_train)

ST6_m_predicted = making_stacking_model12.predict(ST6_test)
ST6_m_predicted[ST6_m_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
ST6_length_x1test = len(ST6_test)
ST6_m_predicted = ST6_m_predicted.reshape(ST6_length_x1test,1)

# 학습 모델 성능 확인
ST6_m_mae = abs(ST6_m_predicted - ST_Y1_test).mean(axis=0)
ST6_m_mape = (np.abs((ST6_m_predicted - ST_Y1_test) / ST_Y1_test).mean(axis=0))
ST6_m_rmse = np.sqrt(((ST6_m_predicted - ST_Y1_test) ** 2).mean(axis=0))
ST6_m_rmsle = np.sqrt((((np.log(ST6_m_predicted + 1) - np.log(ST_Y1_test + 1)) ** 2).mean(axis=0)))

print(ST6_m_mae)
print(ST6_m_mape)
print(ST6_m_rmse)
print(ST6_m_rmsle)

########################################################################################################################
# 분석결과 저장
ST_m_evaluation = {'MAE' : [ST2_m_mae[0],ST6_m_mae[0]], 'MAPE' :  [ST2_m_mape[0],ST6_m_mape[0]], 'RMSE' :  [ST2_m_rmse[0],ST6_m_rmse[0]], 'RMSLE' : [ST2_m_rmsle[0],ST6_m_rmsle[0]]}
ST_m_evaluation = pd.DataFrame(ST_m_evaluation, index = ['test1','test2'])
print(ST_m_evaluation)

# 분석결과 .csv 파일 저장
ST_m_evaluation.to_csv('spool_m_stacking_conclusion.csv', sep=',', na_rep='NaN')