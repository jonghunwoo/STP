# 해당 파일은 기계학습의 다중 선형회귀분석 알고리즘을 적용한 학습모델을 구축하여 성능 확인을 수행
# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

########################################### MakingLT ###################################################################
# Data 불러오기
ri_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del ri_MakingLT['Unnamed: 0']

# Data Type 변경
ri_MakingLT["Emergency"] = ri_MakingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
ri_m_Emergency_one_hot_encoded = pd.get_dummies(ri_MakingLT.Emergency)
ri_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(ri_MakingLT.ApplyLeadTime)
ri_m_STG_one_hot_encoded = pd.get_dummies(ri_MakingLT.STG)
ri_m_Service_one_hot_encoded = pd.get_dummies(ri_MakingLT.Service)
ri_m_Pass_one_hot_encoded = pd.get_dummies(ri_MakingLT.Pass)
ri_m_Sch_one_hot_encoded = pd.get_dummies(ri_MakingLT.Sch)
ri_m_Material_one_hot_encoded = pd.get_dummies(ri_MakingLT.Material)
ri_m_Making_Co_one_hot_encoded = pd.get_dummies(ri_MakingLT.Making_Co)

# Input, Output 분류
ri_m_Inputdata = pd.concat((ri_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , ri_m_Emergency_one_hot_encoded
                      , ri_m_ApplyLeadTime_one_hot_encoded
                      , ri_m_STG_one_hot_encoded
                      , ri_m_Service_one_hot_encoded
                      , ri_m_Pass_one_hot_encoded
                      , ri_m_Sch_one_hot_encoded
                      , ri_m_Material_one_hot_encoded
                      , ri_m_Making_Co_one_hot_encoded)
                      , axis=1)

ri_m_Outputdata = ri_MakingLT[['MakingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
ri_X1 = ri_m_Inputdata.values
ri_Y1 = ri_m_Outputdata.values

# Training Data, Test Data 분리
ri_X1_train, ri_X1_test, ri_Y1_train, ri_Y1_test = train_test_split(ri_X1, ri_Y1, test_size = 0.33, random_state = 42)

########################################################################################################################
# 다중 선형회귀분석 학습 모델 구축
making_regression_model = linear_model.LinearRegression()

making_regression_model.fit(ri_X1_train, ri_Y1_train)

ri_m_predicted = making_regression_model.predict(ri_X1_test)
ri_m_predicted[ri_m_predicted<0] = 0

# 학습 모델 성능 확인
ri_m_mae = abs(ri_m_predicted - ri_Y1_test).mean(axis=0)
ri_m_mape = (np.abs((ri_m_predicted - ri_Y1_test) / ri_Y1_test).mean(axis=0))
ri_m_rmse = np.sqrt(((ri_m_predicted - ri_Y1_test) ** 2).mean(axis=0))
ri_m_rmsle = np.sqrt((((np.log(ri_m_predicted + 1) - np.log(ri_Y1_test + 1)) ** 2).mean(axis=0)))

print(ri_m_mae)
print(ri_m_mape)
print(ri_m_rmse)
print(ri_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
ri_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del ri_PaintingLT['Unnamed: 0']

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
ri_p_Emergency_one_hot_encoded = pd.get_dummies(ri_PaintingLT.Emergency)
ri_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(ri_PaintingLT.ApplyLeadTime)
ri_p_STG_one_hot_encoded = pd.get_dummies(ri_PaintingLT.STG)
ri_p_Service_one_hot_encoded = pd.get_dummies(ri_PaintingLT.Service)
ri_p_Pass_one_hot_encoded = pd.get_dummies(ri_PaintingLT.Pass)
ri_p_Sch_one_hot_encoded = pd.get_dummies(ri_PaintingLT.Sch)
ri_p_Material_one_hot_encoded = pd.get_dummies(ri_PaintingLT.Material)
ri_p_Making_Co_one_hot_encoded = pd.get_dummies(ri_PaintingLT.Making_Co)
ri_p_After2_Co_one_hot_encoded = pd.get_dummies(ri_PaintingLT.After2_Co)

# Input, Output 분류
ri_p_Inputdata = pd.concat((ri_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , ri_p_Emergency_one_hot_encoded
                      , ri_p_ApplyLeadTime_one_hot_encoded
                      , ri_p_STG_one_hot_encoded
                      , ri_p_Service_one_hot_encoded
                      , ri_p_Pass_one_hot_encoded
                      , ri_p_Sch_one_hot_encoded
                      , ri_p_Material_one_hot_encoded
                      , ri_p_Making_Co_one_hot_encoded
                      , ri_p_After2_Co_one_hot_encoded)
                      , axis=1)

ri_p_Outputdata = ri_PaintingLT[['PaintingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
ri_X2 = ri_p_Inputdata.values
ri_Y2 = ri_p_Outputdata.values

# Training Data, Test Data 분리
ri_X2_train, ri_X2_test, ri_Y2_train, ri_Y2_test = train_test_split(ri_X2, ri_Y2, test_size = 0.33, random_state = 42)

########################################################################################################################
# 다중 선형회귀분석 학습 모델 구축
painting_regression_model = linear_model.LinearRegression()

painting_regression_model.fit(ri_X2_train, ri_Y2_train)

ri_p_predicted = painting_regression_model.predict(ri_X2_test)
ri_p_predicted[ri_p_predicted<0] = 0

# 학습 모델 성능 확인
ri_p_mae = abs(ri_p_predicted - ri_Y2_test).mean(axis=0)
ri_p_mape = (np.abs((ri_p_predicted - ri_Y2_test) / ri_Y2_test).mean(axis=0))
ri_p_rmse = np.sqrt(((ri_p_predicted - ri_Y2_test) ** 2).mean(axis=0))
ri_p_rmsle = np.sqrt((((np.log(ri_p_predicted + 1) - np.log(ri_Y2_test + 1)) ** 2).mean(axis=0)))

print(ri_p_mae)
print(ri_p_mape)
print(ri_p_rmse)
print(ri_p_rmsle)

########################################################################################################################
# 분석결과 저장
ri_evaluation = {'MAE' :  [ri_m_mae[0], ri_p_mae[0]], 'MAPE' :  [ri_m_mape[0], ri_p_mape[0]], 'RMSE' :  [ri_m_rmse[0], ri_p_rmse[0]], 'RMSLE' : [ri_m_rmsle[0], ri_p_rmsle[0]]}
ri_evaluation = pd.DataFrame(ri_evaluation, index = ['making_regression','painting_regression'])
print(ri_evaluation)

# 분석결과 .csv 파일 저장
ri_evaluation.to_csv('spool_regression_conclusion.csv', sep=',', na_rep='NaN')
