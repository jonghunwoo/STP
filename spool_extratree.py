# 해당 파일은 앙상블학습의 extra trees 알고리즘을 적용한 학습모델을 구축하여 성능 확인을 수행
# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor

########################################### MakingLT ###################################################################
# Data 불러오기
et_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del et_MakingLT['Unnamed: 0']

# Data Type 변경
et_MakingLT["Emergency"] = et_MakingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
et_m_Emergency_one_hot_encoded = pd.get_dummies(et_MakingLT.Emergency)
et_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(et_MakingLT.ApplyLeadTime)
et_m_STG_one_hot_encoded = pd.get_dummies(et_MakingLT.STG)
et_m_Service_one_hot_encoded = pd.get_dummies(et_MakingLT.Service)
et_m_Pass_one_hot_encoded = pd.get_dummies(et_MakingLT.Pass)
et_m_Sch_one_hot_encoded = pd.get_dummies(et_MakingLT.Sch)
et_m_Material_one_hot_encoded = pd.get_dummies(et_MakingLT.Material)
et_m_Making_Co_one_hot_encoded = pd.get_dummies(et_MakingLT.Making_Co)

# Input, Output 분류
et_m_Inputdata = pd.concat((et_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , et_m_Emergency_one_hot_encoded
                      , et_m_ApplyLeadTime_one_hot_encoded
                      , et_m_STG_one_hot_encoded
                      , et_m_Service_one_hot_encoded
                      , et_m_Pass_one_hot_encoded
                      , et_m_Sch_one_hot_encoded
                      , et_m_Material_one_hot_encoded
                      , et_m_Making_Co_one_hot_encoded)
                      , axis=1)

et_m_Outputdata = et_MakingLT[['MakingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
et_X1 = et_m_Inputdata.values
et_Y1 = et_m_Outputdata.values

# Training Data, Test Data 분리
et_X1_train, et_X1_test, et_Y1_train, et_Y1_test = train_test_split(et_X1, et_Y1, test_size = 0.33, random_state = 42)

########################################################################################################################
# ExtraTree 모델 구축
making_extratree_model = ExtraTreeRegressor(max_depth=10, random_state=42)

making_extratree_model.fit(et_X1_train, et_Y1_train)

et_m_predicted = making_extratree_model.predict(et_X1_test)
et_m_predicted[et_m_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
et_length_x1test = len(et_X1_test)
et_m_predicted = et_m_predicted.reshape(et_length_x1test,1)

# 학습 모델 성능 확인
et_m_mae = abs(et_m_predicted - et_Y1_test).mean(axis=0)
et_m_mape = (np.abs((et_m_predicted - et_Y1_test) / et_Y1_test).mean(axis=0))
et_m_rmse = np.sqrt(((et_m_predicted - et_Y1_test) ** 2).mean(axis=0))
et_m_rmsle = np.sqrt((((np.log(et_m_predicted + 1) - np.log(et_Y1_test + 1)) ** 2).mean(axis=0)))

print(et_m_mae)
print(et_m_mape)
print(et_m_rmse)
print(et_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
et_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del et_PaintingLT['Unnamed: 0']

# Data Type 변경
et_PaintingLT["Emergency"] = et_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
et_p_Emergency_one_hot_encoded = pd.get_dummies(et_PaintingLT.Emergency)
et_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(et_PaintingLT.ApplyLeadTime)
et_p_STG_one_hot_encoded = pd.get_dummies(et_PaintingLT.STG)
et_p_Service_one_hot_encoded = pd.get_dummies(et_PaintingLT.Service)
et_p_Pass_one_hot_encoded = pd.get_dummies(et_PaintingLT.Pass)
et_p_Sch_one_hot_encoded = pd.get_dummies(et_PaintingLT.Sch)
et_p_Material_one_hot_encoded = pd.get_dummies(et_PaintingLT.Material)
et_p_Making_Co_one_hot_encoded = pd.get_dummies(et_PaintingLT.Making_Co)
et_p_After2_Co_one_hot_encoded = pd.get_dummies(et_PaintingLT.After2_Co)

# Input, Output 분류
et_p_Inputdata = pd.concat((et_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , et_p_Emergency_one_hot_encoded
                      , et_p_ApplyLeadTime_one_hot_encoded
                      , et_p_STG_one_hot_encoded
                      , et_p_Service_one_hot_encoded
                      , et_p_Pass_one_hot_encoded
                      , et_p_Sch_one_hot_encoded
                      , et_p_Material_one_hot_encoded
                      , et_p_Making_Co_one_hot_encoded
                      , et_p_After2_Co_one_hot_encoded)
                      , axis=1)

et_p_Outputdata = et_PaintingLT[['PaintingLT']]

# 학습모델 구축을 위해 data형식을 Vector로 변환
et_X2 = et_p_Inputdata.values
et_Y2 = et_p_Outputdata.values

# Training Data, Test Data 분리
et_X2_train, et_X2_test, et_Y2_train, et_Y2_test = train_test_split(et_X2, et_Y2, test_size = 0.33, random_state = 42)

########################################################################################################################
# ExtraTree 모델 구축
painting_extratree_model = ExtraTreeRegressor(max_depth=10, random_state=42)

painting_extratree_model.fit(et_X2_train, et_Y2_train)

et_p_predicted = painting_extratree_model.predict(et_X2_test)
et_p_predicted[et_p_predicted<0] = 0

# [1,n]에서 [n,1]로 배열을 바꿔주는 과정을 추가
et_length_x2test = len(et_X2_test)
et_p_predicted = et_p_predicted.reshape(et_length_x2test,1)

# 학습 모델 성능 확인
et_p_mae = abs(et_p_predicted - et_Y2_test).mean(axis=0)
et_p_mape = (np.abs((et_p_predicted - et_Y2_test) / et_Y2_test).mean(axis=0))
et_p_rmse = np.sqrt(((et_p_predicted - et_Y2_test) ** 2).mean(axis=0))
et_p_rmsle = np.sqrt((((np.log(et_p_predicted + 1) - np.log(et_Y2_test + 1)) ** 2).mean(axis=0)))

print(et_p_mae)
print(et_p_mape)
print(et_p_rmse)
print(et_p_rmsle)

########################################################################################################################
# 분석결과 저장
et_evaluation = {'MAE' :  [et_m_mae[0], et_p_mae[0]], 'MAPE' :  [et_m_mape[0], et_p_mape[0]], 'RMSE' :  [et_m_rmse[0], et_p_rmse[0]], 'RMSLE' : [et_m_rmsle[0], et_p_rmsle[0]]}
et_evaluation = pd.DataFrame(et_evaluation, index = ['making_extratree','painting_extratree'])
print(et_evaluation)

# 분석결과 .csv 파일 저장
et_evaluation.to_csv('spool_extratree_conclusion.csv', sep=',', na_rep='NaN')