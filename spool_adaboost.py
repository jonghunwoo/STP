# 해당 파일은 앙상블학습의 adaboost 알고리즘을 적용한 학습모델을 구축하여 성능 확인을 수행
# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

########################################### MakingLT ###################################################################
# Data 불러오기
AB_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del AB_MakingLT['Unnamed: 0']

# Data Type 변경
AB_MakingLT["Emergency"] = AB_MakingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
AB_m_Emergency_one_hot_encoded = pd.get_dummies(AB_MakingLT.Emergency)
AB_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(AB_MakingLT.ApplyLeadTime)
AB_m_STG_one_hot_encoded = pd.get_dummies(AB_MakingLT.STG)
AB_m_Service_one_hot_encoded = pd.get_dummies(AB_MakingLT.Service)
AB_m_Pass_one_hot_encoded = pd.get_dummies(AB_MakingLT.Pass)
AB_m_Sch_one_hot_encoded = pd.get_dummies(AB_MakingLT.Sch)
AB_m_Material_one_hot_encoded = pd.get_dummies(AB_MakingLT.Material)
AB_m_Making_Co_one_hot_encoded = pd.get_dummies(AB_MakingLT.Making_Co)

# Input, Output 분류
AB_m_Inputdata = pd.concat((AB_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , AB_m_Emergency_one_hot_encoded
                      , AB_m_ApplyLeadTime_one_hot_encoded
                      , AB_m_STG_one_hot_encoded
                      , AB_m_Service_one_hot_encoded
                      , AB_m_Pass_one_hot_encoded
                      , AB_m_Sch_one_hot_encoded
                      , AB_m_Material_one_hot_encoded
                      , AB_m_Making_Co_one_hot_encoded)
                      , axis=1)

AB_m_Outputdata = AB_MakingLT[['MakingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
AB_X1 = AB_m_Inputdata.values
AB_Y1 = AB_m_Outputdata.values

# Training Data, Test Data 분리
AB_X1_train, AB_X1_test, AB_Y1_train, AB_Y1_test = train_test_split(AB_X1, AB_Y1, test_size = 0.33, random_state = 42)

########################################################################################################################
# AdaBoost 학습 모델 구축
making_adaboost_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=100, learning_rate=0.5, random_state=42)

making_adaboost_model.fit(AB_X1_train, AB_Y1_train)

AB_m_predicted = making_adaboost_model.predict(AB_X1_test)

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
AB_length_x1test = len(AB_X1_test)
AB_m_predicted = AB_m_predicted.reshape(AB_length_x1test,1)

# 학습 모델 성능 확인
AB_m_mae = abs(AB_m_predicted - AB_Y1_test).mean(axis=0)
AB_m_mape = (np.abs((AB_m_predicted - AB_Y1_test) / AB_Y1_test).mean(axis=0))
AB_m_rmse = np.sqrt(((AB_m_predicted - AB_Y1_test) ** 2).mean(axis=0))
AB_m_rmsle = np.sqrt((((np.log(AB_m_predicted + 1) - np.log(AB_Y1_test + 1)) ** 2).mean(axis=0)))

print(AB_m_mae)
print(AB_m_mape)
print(AB_m_rmse)
print(AB_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
AB_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del AB_PaintingLT['Unnamed: 0']

# Data Type 변경
AB_PaintingLT["Emergency"] = AB_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
AB_p_Emergency_one_hot_encoded = pd.get_dummies(AB_PaintingLT.Emergency)
AB_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(AB_PaintingLT.ApplyLeadTime)
AB_p_STG_one_hot_encoded = pd.get_dummies(AB_PaintingLT.STG)
AB_p_Service_one_hot_encoded = pd.get_dummies(AB_PaintingLT.Service)
AB_p_Pass_one_hot_encoded = pd.get_dummies(AB_PaintingLT.Pass)
AB_p_Sch_one_hot_encoded = pd.get_dummies(AB_PaintingLT.Sch)
AB_p_Material_one_hot_encoded = pd.get_dummies(AB_PaintingLT.Material)
AB_p_Making_Co_one_hot_encoded = pd.get_dummies(AB_PaintingLT.Making_Co)
AB_p_After2_Co_one_hot_encoded = pd.get_dummies(AB_PaintingLT.After2_Co)

# Input, Output 분류
AB_p_Inputdata = pd.concat((AB_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , AB_p_Emergency_one_hot_encoded
                      , AB_p_ApplyLeadTime_one_hot_encoded
                      , AB_p_STG_one_hot_encoded
                      , AB_p_Service_one_hot_encoded
                      , AB_p_Pass_one_hot_encoded
                      , AB_p_Sch_one_hot_encoded
                      , AB_p_Material_one_hot_encoded
                      , AB_p_Making_Co_one_hot_encoded
                      , AB_p_After2_Co_one_hot_encoded)
                      , axis=1)

AB_p_Outputdata = AB_PaintingLT[['PaintingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
AB_X2 = AB_p_Inputdata.values
AB_Y2 = AB_p_Outputdata.values

# Training Data, Test Data 분리
AB_X2_train, AB_X2_test, AB_Y2_train, AB_Y2_test = train_test_split(AB_X2, AB_Y2, test_size = 0.33, random_state = 42)

########################################################################################################################
# AdaBoost 모델 구축
painting_adaboost_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=100, learning_rate=0.5, random_state=42)

painting_adaboost_model.fit(AB_X2_train, AB_Y2_train)

AB_p_predicted = painting_adaboost_model.predict(AB_X2_test)
AB_p_predicted[AB_p_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
AB_length_x2test = len(AB_X2_test)
AB_p_predicted = AB_p_predicted.reshape(AB_length_x2test,1)

# 학습 모델 성능 확인
AB_p_mae = abs(AB_p_predicted - AB_Y2_test).mean(axis=0)
AB_p_mape = (np.abs((AB_p_predicted - AB_Y2_test) / AB_Y2_test).mean(axis=0))
AB_p_rmse = np.sqrt(((AB_p_predicted - AB_Y2_test) ** 2).mean(axis=0))
AB_p_rmsle = np.sqrt((((np.log(AB_p_predicted + 1) - np.log(AB_Y2_test + 1)) ** 2).mean(axis=0)))

print(AB_p_mae)
print(AB_p_mape)
print(AB_p_rmse)
print(AB_p_rmsle)

########################################################################################################################
# 분석결과 저장
AB_evaluation = {'MAE' :  [AB_m_mae[0], AB_p_mae[0]], 'MAPE' :  [AB_m_mape[0], AB_p_mape[0]], 'RMSE' :  [AB_m_rmse[0], AB_p_rmse[0]], 'RMSLE' : [AB_m_rmsle[0], AB_p_rmsle[0]]}
AB_evaluation = pd.DataFrame(AB_evaluation, index = ['making_adaboost','painting_adaboost'])
print(AB_evaluation)

# 분석결과 .csv 파일 저장
AB_evaluation.to_csv('spool_adaboost_conclusion.csv', sep=',', na_rep='NaN')