# Package Load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

########################################### MakingLT ###################################################################
# Data 불러오기
dt_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del dt_MakingLT['Unnamed: 0']

# Data Type 변경
dt_MakingLT["Emergency"] = dt_MakingLT["Emergency"].astype(np.object)

# One-hot encoding
dt_m_Emergency_one_hot_encoded = pd.get_dummies(dt_MakingLT.Emergency)
dt_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(dt_MakingLT.ApplyLeadTime)
dt_m_STG_one_hot_encoded = pd.get_dummies(dt_MakingLT.STG)
dt_m_Service_one_hot_encoded = pd.get_dummies(dt_MakingLT.Service)
dt_m_Pass_one_hot_encoded = pd.get_dummies(dt_MakingLT.Pass)
dt_m_Sch_one_hot_encoded = pd.get_dummies(dt_MakingLT.Sch)
dt_m_Material_one_hot_encoded = pd.get_dummies(dt_MakingLT.Material)
dt_m_Making_Co_one_hot_encoded = pd.get_dummies(dt_MakingLT.Making_Co)

# Input, Output 분류
dt_m_Inputdata = pd.concat((dt_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , dt_m_Emergency_one_hot_encoded
                      , dt_m_ApplyLeadTime_one_hot_encoded
                      , dt_m_STG_one_hot_encoded
                      , dt_m_Service_one_hot_encoded
                      , dt_m_Pass_one_hot_encoded
                      , dt_m_Sch_one_hot_encoded
                      , dt_m_Material_one_hot_encoded
                      , dt_m_Making_Co_one_hot_encoded)
                      , axis=1)

dt_m_Outputdata = dt_MakingLT[['MakingLT']]

# Vector로 변환 ; Deeplearning에서는 Vector 사용
dt_X1 = dt_m_Inputdata.values
dt_Y1 = dt_m_Outputdata.values

# Training Data, Test Data 분리
dt_X1_train, dt_X1_test, dt_Y1_train, dt_Y1_test = train_test_split(dt_X1, dt_Y1, test_size = 0.33, random_state = 42)

# 의사결정나무 모델 구축
making_decisiontree_model = DecisionTreeClassifier(max_depth=20, random_state=42)

making_decisiontree_model.fit(dt_X1_train, dt_Y1_train)

dt_m_predicted = making_decisiontree_model.predict(dt_X1_test)

dt_length_x1test = len(dt_X1_test)
dt_m_predicted = dt_m_predicted.reshape(dt_length_x1test,1)

dt_m_mae = abs(dt_m_predicted - dt_Y1_test).mean(axis=0)
dt_m_mape = (np.abs((dt_m_predicted - dt_Y1_test) / dt_Y1_test).mean(axis=0))
dt_m_rmse = np.sqrt(((dt_m_predicted - dt_Y1_test) ** 2).mean(axis=0))
dt_m_rmsle = np.sqrt((((np.log(dt_m_predicted + 1) - np.log(dt_Y1_test + 1)) ** 2).mean(axis=0)))

print(dt_m_mae)
print(dt_m_mape)
print(dt_m_rmse)
print(dt_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
dt_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del dt_PaintingLT['Unnamed: 0']

# Data Type 변경
dt_PaintingLT["Emergency"] = dt_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding
dt_p_Emergency_one_hot_encoded = pd.get_dummies(dt_PaintingLT.Emergency)
dt_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(dt_PaintingLT.ApplyLeadTime)
dt_p_STG_one_hot_encoded = pd.get_dummies(dt_PaintingLT.STG)
dt_p_Service_one_hot_encoded = pd.get_dummies(dt_PaintingLT.Service)
dt_p_Pass_one_hot_encoded = pd.get_dummies(dt_PaintingLT.Pass)
dt_p_Sch_one_hot_encoded = pd.get_dummies(dt_PaintingLT.Sch)
dt_p_Material_one_hot_encoded = pd.get_dummies(dt_PaintingLT.Material)
dt_p_Making_Co_one_hot_encoded = pd.get_dummies(dt_PaintingLT.Making_Co)
dt_p_After2_Co_one_hot_encoded = pd.get_dummies(dt_PaintingLT.After2_Co)

# Input, Output 분류
dt_p_Inputdata = pd.concat((dt_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , dt_p_Emergency_one_hot_encoded
                      , dt_p_ApplyLeadTime_one_hot_encoded
                      , dt_p_STG_one_hot_encoded
                      , dt_p_Service_one_hot_encoded
                      , dt_p_Pass_one_hot_encoded
                      , dt_p_Sch_one_hot_encoded
                      , dt_p_Material_one_hot_encoded
                      , dt_p_Making_Co_one_hot_encoded
                      , dt_p_After2_Co_one_hot_encoded)
                      , axis=1)

dt_p_Outputdata = dt_PaintingLT[['PaintingLT']]

# 'Vector'로 변환 ; Deeplearning에서는 Vector값이 사용
dt_X2 = dt_p_Inputdata.values
dt_Y2 = dt_p_Outputdata.values

# Training Data, Test Data 분리
dt_X2_train, dt_X2_test, dt_Y2_train, dt_Y2_test = train_test_split(dt_X2, dt_Y2, test_size = 0.33, random_state = 42)

# 의사결정나무 모델 구축
painting_decisiontree_model = DecisionTreeClassifier(max_depth=20, random_state=42)

painting_decisiontree_model.fit(dt_X2_train, dt_Y2_train)

dt_p_predicted = painting_decisiontree_model.predict(dt_X2_test)

dt_length_x2test = len(dt_X2_test)
dt_p_predicted = dt_p_predicted.reshape(dt_length_x2test,1)

dt_p_mae = abs(dt_p_predicted - dt_Y2_test).mean(axis=0)
dt_p_mape = (np.abs((dt_p_predicted - dt_Y2_test) / dt_Y2_test).mean(axis=0))
dt_p_rmse = np.sqrt(((dt_p_predicted - dt_Y2_test) ** 2).mean(axis=0))
dt_p_rmsle = np.sqrt((((np.log(dt_p_predicted + 1) - np.log(dt_Y2_test + 1)) ** 2).mean(axis=0)))

print(dt_p_mae)
print(dt_p_mape)
print(dt_p_rmse)
print(dt_p_rmsle)

########################################################################################################################
# 분석결과 저장
dt_evaluation = {'MAE' :  [dt_m_mae[0], dt_p_mae[0]], 'MAPE' :  [dt_m_mape[0], dt_p_mape[0]], 'RMSE' :  [dt_m_rmse[0], dt_p_rmse[0]], 'RMSLE' : [dt_m_rmsle[0], dt_p_rmsle[0]]}

dt_evaluation = pd.DataFrame(dt_evaluation, index = ['making_decisiontree','painting_decisiontree'])

print(dt_evaluation)

# 분석결과 .csv 파일 저장
dt_evaluation.to_csv('spool_decisiontree_conclusion.csv', sep=',', na_rep='NaN')
