# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost

########################################### MakingLT ###################################################################
# Data 불러오기
XG_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del XG_MakingLT['Unnamed: 0']

# Data Type 변경
XG_MakingLT["Emergency"] = XG_MakingLT["Emergency"].astype(np.object)

# One-hot encoding
XG_m_Emergency_one_hot_encoded = pd.get_dummies(XG_MakingLT.Emergency)
XG_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(XG_MakingLT.ApplyLeadTime)
XG_m_STG_one_hot_encoded = pd.get_dummies(XG_MakingLT.STG)
XG_m_Service_one_hot_encoded = pd.get_dummies(XG_MakingLT.Service)
XG_m_Pass_one_hot_encoded = pd.get_dummies(XG_MakingLT.Pass)
XG_m_Sch_one_hot_encoded = pd.get_dummies(XG_MakingLT.Sch)
XG_m_Material_one_hot_encoded = pd.get_dummies(XG_MakingLT.Material)
XG_m_Making_Co_one_hot_encoded = pd.get_dummies(XG_MakingLT.Making_Co)

# Input, Output 분류
XG_m_Inputdata = pd.concat((XG_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , XG_m_Emergency_one_hot_encoded
                      , XG_m_ApplyLeadTime_one_hot_encoded
                      , XG_m_STG_one_hot_encoded
                      , XG_m_Service_one_hot_encoded
                      , XG_m_Pass_one_hot_encoded
                      , XG_m_Sch_one_hot_encoded
                      , XG_m_Material_one_hot_encoded
                      , XG_m_Making_Co_one_hot_encoded)
                      , axis=1)

XG_m_Outputdata = XG_MakingLT[['MakingLT']]

# Vector로 변환 ; Deeplearning에서는 Vector 사용
XG_X1 = XG_m_Inputdata.values
XG_Y1 = XG_m_Outputdata.values

# Training Data, Test Data 분리
XG_X1_train, XG_X1_test, XG_Y1_train, XG_Y1_test = train_test_split(XG_X1, XG_Y1, test_size = 0.33, random_state = 42)

# XGBoost 모델 구축
making_xgboost_model = xgboost.XGBRegressor(colsample_bytree=0.5, learning_rate=0.5, max_depth=20, n_estimators=100)

making_xgboost_model.fit(XG_X1_train, XG_Y1_train)

XG_m_predicted = making_xgboost_model.predict(XG_X1_test)

XG_length_x1test = len(XG_X1_test)
XG_m_predicted = XG_m_predicted.reshape(XG_length_x1test,1)

XG_m_mae = abs(XG_m_predicted - XG_Y1_test).mean(axis=0)
XG_m_mape = (np.abs((XG_m_predicted - XG_Y1_test) / XG_Y1_test).mean(axis=0))
XG_m_rmse = np.sqrt(((XG_m_predicted - XG_Y1_test) ** 2).mean(axis=0))
XG_m_rmsle = np.sqrt((((np.log(XG_m_predicted + 1) - np.log(XG_Y1_test + 1)) ** 2).mean(axis=0)))

print(XG_m_mae)
print(XG_m_mape)
print(XG_m_rmse)
print(XG_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
XG_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del XG_PaintingLT['Unnamed: 0']

# One-hot encoding
XG_p_Emergency_one_hot_encoded = pd.get_dummies(XG_PaintingLT.Emergency)
XG_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(XG_PaintingLT.ApplyLeadTime)
XG_p_STG_one_hot_encoded = pd.get_dummies(XG_PaintingLT.STG)
XG_p_Service_one_hot_encoded = pd.get_dummies(XG_PaintingLT.Service)
XG_p_Pass_one_hot_encoded = pd.get_dummies(XG_PaintingLT.Pass)
XG_p_Sch_one_hot_encoded = pd.get_dummies(XG_PaintingLT.Sch)
XG_p_Material_one_hot_encoded = pd.get_dummies(XG_PaintingLT.Material)
XG_p_Making_Co_one_hot_encoded = pd.get_dummies(XG_PaintingLT.Making_Co)
XG_p_After2_Co_one_hot_encoded = pd.get_dummies(XG_PaintingLT.After2_Co)

# Input, Output 분류
XG_p_Inputdata = pd.concat((XG_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , XG_p_Emergency_one_hot_encoded
                      , XG_p_ApplyLeadTime_one_hot_encoded
                      , XG_p_STG_one_hot_encoded
                      , XG_p_Service_one_hot_encoded
                      , XG_p_Pass_one_hot_encoded
                      , XG_p_Sch_one_hot_encoded
                      , XG_p_Material_one_hot_encoded
                      , XG_p_Making_Co_one_hot_encoded
                      , XG_p_After2_Co_one_hot_encoded)
                      , axis=1)

XG_p_Outputdata = XG_PaintingLT[['PaintingLT']]

# Vector로 변환 ; Deeplearning에서는 Vector 사용
XG_X2 = XG_p_Inputdata.values
XG_Y2 = XG_p_Outputdata.values

# Training Data, Test Data 분리
XG_X2_train, XG_X2_test, XG_Y2_train, XG_Y2_test = train_test_split(XG_X2, XG_Y2, test_size = 0.33, random_state = 42)

# GradientBoost 모델 구축
painting_xgboost_model = xgboost.XGBRegressor(colsample_bytree=0.5, learning_rate=0.5, max_depth=20, n_estimators=100)

painting_xgboost_model.fit(XG_X2_train, XG_Y2_train)

XG_p_predicted = painting_xgboost_model.predict(XG_X2_test)

XG_length_x2test = len(XG_X2_test)
XG_p_predicted = XG_p_predicted.reshape(XG_length_x2test,1)

XG_p_mae = abs(XG_p_predicted - XG_Y2_test).mean(axis=0)
XG_p_mape = (np.abs((XG_p_predicted - XG_Y2_test) / XG_Y2_test).mean(axis=0))
XG_p_rmse = np.sqrt(((XG_p_predicted - XG_Y2_test) ** 2).mean(axis=0))
XG_p_rmsle = np.sqrt((((np.log(XG_p_predicted + 1) - np.log(XG_Y2_test + 1)) ** 2).mean(axis=0)))

print(XG_p_mae)
print(XG_p_mape)
print(XG_p_rmse)
print(XG_p_rmsle)

########################################################################################################################
# 분석결과 저장
XG_evaluation = {'MAE' :  [XG_m_mae[0], XG_p_mae[0]], 'MAPE' :  [XG_m_mape[0], XG_p_mape[0]], 'RMSE' :  [XG_m_rmse[0], XG_p_rmse[0]], 'RMSLE' : [XG_m_rmsle[0], XG_p_rmsle[0]]}

XG_evaluation = pd.DataFrame(XG_evaluation, index = ['making_xgboost','painting_xgboost'])

print(XG_evaluation)

# 분석결과 .csv 파일 저장
XG_evaluation.to_csv('spool_xgboost_conclusion.csv', sep=',', na_rep='NaN')