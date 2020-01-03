# 해당 파일은 앙상블학습의 randomforest 알고리즘을 적용한 학습모델을 구축하여 성능 확인을 수행
# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

########################################### MakingLT ###################################################################
# Data 불러오기
rf_MakingLT = pd.read_csv('./data/p_makingdata.csv', encoding='euc-kr')

# 첫번째 Column 삭제
del rf_MakingLT['Unnamed: 0']

# Data Type 변경
rf_MakingLT["Emergency"] = rf_MakingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
rf_m_Emergency_one_hot_encoded = pd.get_dummies(rf_MakingLT.Emergency)
rf_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(rf_MakingLT.ApplyLeadTime)
rf_m_STG_one_hot_encoded = pd.get_dummies(rf_MakingLT.STG)
rf_m_Service_one_hot_encoded = pd.get_dummies(rf_MakingLT.Service)
rf_m_Pass_one_hot_encoded = pd.get_dummies(rf_MakingLT.Pass)
rf_m_Sch_one_hot_encoded = pd.get_dummies(rf_MakingLT.Sch)
rf_m_Material_one_hot_encoded = pd.get_dummies(rf_MakingLT.Material)
rf_m_Making_Co_one_hot_encoded = pd.get_dummies(rf_MakingLT.Making_Co)

# Input, Output 분류
rf_m_Inputdata = pd.concat((rf_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , rf_m_Emergency_one_hot_encoded
                      , rf_m_ApplyLeadTime_one_hot_encoded
                      , rf_m_STG_one_hot_encoded
                      , rf_m_Service_one_hot_encoded
                      , rf_m_Pass_one_hot_encoded
                      , rf_m_Sch_one_hot_encoded
                      , rf_m_Material_one_hot_encoded
                      , rf_m_Making_Co_one_hot_encoded)
                      , axis=1)

rf_m_Outputdata = rf_MakingLT[['MakingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
rf_X1 = rf_m_Inputdata.values
rf_Y1 = rf_m_Outputdata.values

# Training Data, Test Data 분리
rf_X1_train, rf_X1_test, rf_Y1_train, rf_Y1_test = train_test_split(rf_X1, rf_Y1, test_size = 0.33, random_state = 42)

########################################################################################################################
# 랜덤포레스트 모델 구축
making_randomforest_model = RandomForestRegressor(max_depth=10, n_estimators=500, random_state=42)

making_randomforest_model.fit(rf_X1_train, rf_Y1_train)

rf_m_predicted = making_randomforest_model.predict(rf_X1_test)
rf_m_predicted[rf_m_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
rf_length_x1test = len(rf_X1_test)
rf_m_predicted = rf_m_predicted.reshape(rf_length_x1test,1)

# 학습 모델 성능 확인
rf_m_mae = abs(rf_m_predicted - rf_Y1_test).mean(axis=0)
rf_m_mape = (np.abs((rf_m_predicted - rf_Y1_test) / rf_Y1_test).mean(axis=0))
rf_m_rmse = np.sqrt(((rf_m_predicted - rf_Y1_test) ** 2).mean(axis=0))
rf_m_rmsle = np.sqrt((((np.log(rf_m_predicted + 1) - np.log(rf_Y1_test + 1)) ** 2).mean(axis=0)))

print(rf_m_mae)
print(rf_m_mape)
print(rf_m_rmse)
print(rf_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
rf_PaintingLT = pd.read_csv('./data/p_paintingdata.csv', encoding='euc-kr')

# 첫번째 Column 삭제
del rf_PaintingLT['Unnamed: 0']

# Data Type 변경
rf_PaintingLT["Emergency"] = rf_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
rf_p_Emergency_one_hot_encoded = pd.get_dummies(rf_PaintingLT.Emergency)
rf_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(rf_PaintingLT.ApplyLeadTime)
rf_p_STG_one_hot_encoded = pd.get_dummies(rf_PaintingLT.STG)
rf_p_Service_one_hot_encoded = pd.get_dummies(rf_PaintingLT.Service)
rf_p_Pass_one_hot_encoded = pd.get_dummies(rf_PaintingLT.Pass)
rf_p_Sch_one_hot_encoded = pd.get_dummies(rf_PaintingLT.Sch)
rf_p_Material_one_hot_encoded = pd.get_dummies(rf_PaintingLT.Material)
rf_p_Making_Co_one_hot_encoded = pd.get_dummies(rf_PaintingLT.Making_Co)
rf_p_After2_Co_one_hot_encoded = pd.get_dummies(rf_PaintingLT.After2_Co)

# Input, Output 분류
rf_p_Inputdata = pd.concat((rf_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , rf_p_Emergency_one_hot_encoded
                      , rf_p_ApplyLeadTime_one_hot_encoded
                      , rf_p_STG_one_hot_encoded
                      , rf_p_Service_one_hot_encoded
                      , rf_p_Pass_one_hot_encoded
                      , rf_p_Sch_one_hot_encoded
                      , rf_p_Material_one_hot_encoded
                      , rf_p_Making_Co_one_hot_encoded
                      , rf_p_After2_Co_one_hot_encoded)
                      , axis=1)

rf_p_Outputdata = rf_PaintingLT[['PaintingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
rf_X2 = rf_p_Inputdata.values
rf_Y2 = rf_p_Outputdata.values

# Training Data, Test Data 분리
rf_X2_train, rf_X2_test, rf_Y2_train, rf_Y2_test = train_test_split(rf_X2, rf_Y2, test_size = 0.33, random_state = 42)

########################################################################################################################
# 랜덤포레스트모델 구축
painting_randomforest_model = RandomForestRegressor(max_depth=10, n_estimators=500, random_state=42)

painting_randomforest_model.fit(rf_X2_train, rf_Y2_train)

rf_p_predicted = painting_randomforest_model.predict(rf_X2_test)
rf_p_predicted[rf_p_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
rf_length_x2test = len(rf_X2_test)
rf_p_predicted = rf_p_predicted.reshape(rf_length_x2test,1)

# 학습 모델 성능 확인
rf_p_mae = abs(rf_p_predicted - rf_Y2_test).mean(axis=0)
rf_p_mape = (np.abs((rf_p_predicted - rf_Y2_test) / rf_Y2_test).mean(axis=0))
rf_p_rmse = np.sqrt(((rf_p_predicted - rf_Y2_test) ** 2).mean(axis=0))
rf_p_rmsle = np.sqrt((((np.log(rf_p_predicted + 1) - np.log(rf_Y2_test + 1)) ** 2).mean(axis=0)))

print(rf_p_mae)
print(rf_p_mape)
print(rf_p_rmse)
print(rf_p_rmsle)

########################################################################################################################
# 분석결과 저장
rf_evaluation = {'MAE' :  [rf_m_mae[0], rf_p_mae[0]], 'MAPE' :  [rf_m_mape[0], rf_p_mape[0]], 'RMSE' :  [rf_m_rmse[0], rf_p_rmse[0]], 'RMSLE' : [rf_m_rmsle[0], rf_p_rmsle[0]]}
rf_evaluation = pd.DataFrame(rf_evaluation, index = ['making_randomforest','painting_randomforest'])
print(rf_evaluation)

# 분석결과 .csv 파일 저장
rf_evaluation.to_csv('./data/spool_randomforest_conclusion.csv', sep=',', na_rep='NaN', encoding='euc-kr')