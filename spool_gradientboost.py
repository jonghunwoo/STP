# 해당 파일은 앙상블학습의 gradient boost 알고리즘을 적용한 학습모델을 구축하여 성능 확인을 수행
# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

########################################### MakingLT ###################################################################
# Data 불러오기
GB_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del GB_MakingLT['Unnamed: 0']

# Data Type 변경
GB_MakingLT["Emergency"] = GB_MakingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
GB_m_Emergency_one_hot_encoded = pd.get_dummies(GB_MakingLT.Emergency)
GB_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(GB_MakingLT.ApplyLeadTime)
GB_m_STG_one_hot_encoded = pd.get_dummies(GB_MakingLT.STG)
GB_m_Service_one_hot_encoded = pd.get_dummies(GB_MakingLT.Service)
GB_m_Pass_one_hot_encoded = pd.get_dummies(GB_MakingLT.Pass)
GB_m_Sch_one_hot_encoded = pd.get_dummies(GB_MakingLT.Sch)
GB_m_Material_one_hot_encoded = pd.get_dummies(GB_MakingLT.Material)
GB_m_Making_Co_one_hot_encoded = pd.get_dummies(GB_MakingLT.Making_Co)

# Input, Output 분류
GB_m_Inputdata = pd.concat((GB_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , GB_m_Emergency_one_hot_encoded
                      , GB_m_ApplyLeadTime_one_hot_encoded
                      , GB_m_STG_one_hot_encoded
                      , GB_m_Service_one_hot_encoded
                      , GB_m_Pass_one_hot_encoded
                      , GB_m_Sch_one_hot_encoded
                      , GB_m_Material_one_hot_encoded
                      , GB_m_Making_Co_one_hot_encoded)
                      , axis=1)

GB_m_Outputdata = GB_MakingLT[['MakingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
GB_X1 = GB_m_Inputdata.values
GB_Y1 = GB_m_Outputdata.values

# Training Data, Test Data 분리
GB_X1_train, GB_X1_test, GB_Y1_train, GB_Y1_test = train_test_split(GB_X1, GB_Y1, test_size = 0.33, random_state = 42)

########################################################################################################################
# GradientBoost 학습 모델 구축
making_gradientboost_model = GradientBoostingRegressor(max_depth=10, n_estimators=100, learning_rate=1, random_state=42)

making_gradientboost_model.fit(GB_X1_train, GB_Y1_train)

GB_m_predicted = making_gradientboost_model.predict(GB_X1_test)
GB_m_predicted[GB_m_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
GB_length_x1test = len(GB_X1_test)
GB_m_predicted = GB_m_predicted.reshape(GB_length_x1test,1)

# 학습 모델 성능 확인
GB_m_mae = abs(GB_m_predicted - GB_Y1_test).mean(axis=0)
GB_m_mape = (np.abs((GB_m_predicted - GB_Y1_test) / GB_Y1_test).mean(axis=0))
GB_m_rmse = np.sqrt(((GB_m_predicted - GB_Y1_test) ** 2).mean(axis=0))
GB_m_rmsle = np.sqrt((((np.log(GB_m_predicted + 1) - np.log(GB_Y1_test + 1)) ** 2).mean(axis=0)))

print(GB_m_mae)
print(GB_m_mape)
print(GB_m_rmse)
print(GB_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
GB_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del GB_PaintingLT['Unnamed: 0']

# Data Type 변경
GB_PaintingLT["Emergency"] = GB_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
GB_p_Emergency_one_hot_encoded = pd.get_dummies(GB_PaintingLT.Emergency)
GB_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(GB_PaintingLT.ApplyLeadTime)
GB_p_STG_one_hot_encoded = pd.get_dummies(GB_PaintingLT.STG)
GB_p_Service_one_hot_encoded = pd.get_dummies(GB_PaintingLT.Service)
GB_p_Pass_one_hot_encoded = pd.get_dummies(GB_PaintingLT.Pass)
GB_p_Sch_one_hot_encoded = pd.get_dummies(GB_PaintingLT.Sch)
GB_p_Material_one_hot_encoded = pd.get_dummies(GB_PaintingLT.Material)
GB_p_Making_Co_one_hot_encoded = pd.get_dummies(GB_PaintingLT.Making_Co)
GB_p_After2_Co_one_hot_encoded = pd.get_dummies(GB_PaintingLT.After2_Co)

# Input, Output 분류
GB_p_Inputdata = pd.concat((GB_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , GB_p_Emergency_one_hot_encoded
                      , GB_p_ApplyLeadTime_one_hot_encoded
                      , GB_p_STG_one_hot_encoded
                      , GB_p_Service_one_hot_encoded
                      , GB_p_Pass_one_hot_encoded
                      , GB_p_Sch_one_hot_encoded
                      , GB_p_Material_one_hot_encoded
                      , GB_p_Making_Co_one_hot_encoded
                      , GB_p_After2_Co_one_hot_encoded)
                      , axis=1)

GB_p_Outputdata = GB_PaintingLT[['PaintingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
GB_X2 = GB_p_Inputdata.values
GB_Y2 = GB_p_Outputdata.values

# Training Data, Test Data 분리
GB_X2_train, GB_X2_test, GB_Y2_train, GB_Y2_test = train_test_split(GB_X2, GB_Y2, test_size = 0.33, random_state = 42)

########################################################################################################################
# GradientBoost 모델 구축
painting_gradientboost_model = GradientBoostingRegressor(max_depth=10, n_estimators=100, learning_rate=1, random_state=42)

painting_gradientboost_model.fit(GB_X2_train, GB_Y2_train)

GB_p_predicted = painting_gradientboost_model.predict(GB_X2_test)
GB_p_predicted[GB_p_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
GB_length_x2test = len(GB_X2_test)
GB_p_predicted = GB_p_predicted.reshape(GB_length_x2test,1)

# 학습 모델 성능 확인
GB_p_mae = abs(GB_p_predicted - GB_Y2_test).mean(axis=0)
GB_p_mape = (np.abs((GB_p_predicted - GB_Y2_test) / GB_Y2_test).mean(axis=0))
GB_p_rmse = np.sqrt(((GB_p_predicted - GB_Y2_test) ** 2).mean(axis=0))
GB_p_rmsle = np.sqrt((((np.log(GB_p_predicted + 1) - np.log(GB_Y2_test + 1)) ** 2).mean(axis=0)))

print(GB_p_mae)
print(GB_p_mape)
print(GB_p_rmse)
print(GB_p_rmsle)

########################################################################################################################
# 분석결과 저장
GB_evaluation = {'MAE' :  [GB_m_mae[0], GB_p_mae[0]], 'MAPE' :  [GB_m_mape[0], GB_p_mape[0]], 'RMSE' :  [GB_m_rmse[0], GB_p_rmse[0]], 'RMSLE' : [GB_m_rmsle[0], GB_p_rmsle[0]]}
GB_evaluation = pd.DataFrame(GB_evaluation, index = ['making_gradientboost','painting_gradientboost'])
print(GB_evaluation)

# 분석결과 .csv 파일 저장
GB_evaluation.to_csv('spool_gradientboost_conclusion.csv', sep=',', na_rep='NaN')