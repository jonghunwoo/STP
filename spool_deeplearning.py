# Package Load
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

########################################### MakingLT ###################################################################
# Data 불러오기
dl_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del dl_MakingLT['Unnamed: 0']

# Data Type 변경
dl_MakingLT["Emergency"] = dl_MakingLT["Emergency"].astype(np.object)

# One-hot encoding
dl_m_Emergency_one_hot_encoded = pd.get_dummies(dl_MakingLT.Emergency)
dl_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(dl_MakingLT.ApplyLeadTime)
dl_m_STG_one_hot_encoded = pd.get_dummies(dl_MakingLT.STG)
dl_m_Service_one_hot_encoded = pd.get_dummies(dl_MakingLT.Service)
dl_m_Pass_one_hot_encoded = pd.get_dummies(dl_MakingLT.Pass)
dl_m_Sch_one_hot_encoded = pd.get_dummies(dl_MakingLT.Sch)
dl_m_Material_one_hot_encoded = pd.get_dummies(dl_MakingLT.Material)
dl_m_Making_Co_one_hot_encoded = pd.get_dummies(dl_MakingLT.Making_Co)

# Input, Output 분류
dl_m_Inputdata = pd.concat((dl_MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , dl_m_Emergency_one_hot_encoded
                      , dl_m_ApplyLeadTime_one_hot_encoded
                      , dl_m_STG_one_hot_encoded
                      , dl_m_Service_one_hot_encoded
                      , dl_m_Pass_one_hot_encoded
                      , dl_m_Sch_one_hot_encoded
                      , dl_m_Material_one_hot_encoded
                      , dl_m_Making_Co_one_hot_encoded)
                      , axis=1)

dl_m_Outputdata = dl_MakingLT[['MakingLT']]

# 'Vector' 로 변환 ; Deeplearning에서는 Vector값이 사용
dl_X1 = dl_m_Inputdata.values

# 데이터 표준화 적용
scaler = MinMaxScaler(feature_range = (0,1))

# Scaling 적용
dl_X1_scale = scaler.fit_transform(dl_X1)

dl_Y1 = dl_m_Outputdata.values

# Training Data, Test Data 분리
dl_X1_train, dl_X1_test, dl_Y1_train, dl_Y1_test = train_test_split(dl_X1_scale, dl_Y1, test_size = 0.33, random_state = 42)

dl_m_col = len(dl_X1_scale[0])

# 딥러닝 모델 구축
# 1. 모델 구성하기
making_deeplearning_model = Sequential()

# Input Layer
making_deeplearning_model.add(Dense(100, input_dim = dl_m_col, activation='relu'))
making_deeplearning_model.add(Dropout(0.3))

# Hidden Layer 1
making_deeplearning_model.add(Dense(100, activation='relu'))
making_deeplearning_model.add(Dropout(0.3))

# Hidden Layer 2
making_deeplearning_model.add(Dense(100, activation='relu'))
making_deeplearning_model.add(Dropout(0.3))

# Hidden Layer 3
making_deeplearning_model.add(Dense(100, activation='relu'))
making_deeplearning_model.add(Dropout(0.3))

# Output Layer
making_deeplearning_model.add(Dense(1))
making_deeplearning_model.summary()

# 2. 모델 학습과정 설정하기
making_deeplearning_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# 3. 모델 학습시키기
dl_m_hist = making_deeplearning_model.fit(dl_X1_train, dl_Y1_train, epochs=200, batch_size=100)

dl_m_predicted = making_deeplearning_model.predict(dl_X1_test)

dl_m_mae = abs(dl_m_predicted - dl_Y1_test).mean(axis=0)
dl_m_mape = (np.abs((dl_m_predicted - dl_Y1_test) / dl_Y1_test).mean(axis=0))
dl_m_rmse = np.sqrt(((dl_m_predicted - dl_Y1_test) ** 2).mean(axis=0))
dl_m_rmsle = np.sqrt((((np.log(dl_m_predicted+1) - np.log(dl_Y1_test+1)) ** 2).mean(axis=0)))

print(dl_m_mae)
print(dl_m_mape)
print(dl_m_rmse)
print(dl_m_rmsle)

# .h5 파일 저장
making_deeplearning_model.save('making_DL_model.h5')

########################################### PaintingLT #################################################################
# Data 불러오기
dl_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del dl_PaintingLT['Unnamed: 0']

# Data Type 변경
dl_PaintingLT["Emergency"] = dl_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding
dl_p_Emergency_one_hot_encoded = pd.get_dummies(dl_PaintingLT.Emergency)
dl_p_ApplyLeadTime_one_hot_encoded = pd.get_dummies(dl_PaintingLT.ApplyLeadTime)
dl_p_STG_one_hot_encoded = pd.get_dummies(dl_PaintingLT.STG)
dl_p_Service_one_hot_encoded = pd.get_dummies(dl_PaintingLT.Service)
dl_p_Pass_one_hot_encoded = pd.get_dummies(dl_PaintingLT.Pass)
dl_p_Sch_one_hot_encoded = pd.get_dummies(dl_PaintingLT.Sch)
dl_p_Material_one_hot_encoded = pd.get_dummies(dl_PaintingLT.Material)
dl_p_Making_Co_one_hot_encoded = pd.get_dummies(dl_PaintingLT.Making_Co)
dl_p_After2_Co_one_hot_encoded = pd.get_dummies(dl_PaintingLT.After2_Co)

# Input, Output 분류
dl_p_Inputdata = pd.concat((dl_PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , dl_p_Emergency_one_hot_encoded
                      , dl_p_ApplyLeadTime_one_hot_encoded
                      , dl_p_STG_one_hot_encoded
                      , dl_p_Service_one_hot_encoded
                      , dl_p_Pass_one_hot_encoded
                      , dl_p_Sch_one_hot_encoded
                      , dl_p_Material_one_hot_encoded
                      , dl_p_Making_Co_one_hot_encoded
                      , dl_p_After2_Co_one_hot_encoded)
                      , axis=1)

dl_p_Outputdata = dl_PaintingLT[['PaintingLT']]

# 'Vector' 로 변환 ; Deeplearning에서는 Vector값이 사용
dl_X2 = dl_p_Inputdata.values

# Scailing 적용
dl_X2_scale = scaler.fit_transform(dl_X2)

dl_Y2 = dl_p_Outputdata.values

# Training Data, Test Data 분리
dl_X2_train, dl_X2_test, dl_Y2_train, dl_Y2_test = train_test_split(dl_X2_scale, dl_Y2, test_size = 0.33, random_state = 42)

dl_p_col = len(dl_X2_scale[0])

# 딥러닝 모델 구축
# 1. 모델 구성하기
painting_deeplearning_model = Sequential()

# Input Layer
painting_deeplearning_model.add(Dense(100, input_dim = dl_p_col, activation='relu'))
painting_deeplearning_model.add(Dropout(0.3))

# Hidden Layer 1
painting_deeplearning_model.add(Dense(100, activation='relu'))
painting_deeplearning_model.add(Dropout(0.3))

# Hidden Layer 2
painting_deeplearning_model.add(Dense(100, activation='relu'))
painting_deeplearning_model.add(Dropout(0.3))

# Hidden Layer 3
painting_deeplearning_model.add(Dense(100, activation='relu'))
painting_deeplearning_model.add(Dropout(0.3))

# Output Layer
painting_deeplearning_model.add(Dense(1))
painting_deeplearning_model.summary()

# 2. 모델 학습과정 설정하기
painting_deeplearning_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# 3. 모델 학습시키기
dl_p_hist = painting_deeplearning_model.fit(dl_X2_train, dl_Y2_train, epochs=200, batch_size=100)

dl_p_predicted = painting_deeplearning_model.predict(dl_X2_test)

dl_p_mae = abs(dl_p_predicted - dl_Y2_test).mean(axis=0)
dl_p_mape = (np.abs((dl_p_predicted - dl_Y2_test) / dl_Y2_test).mean(axis=0))
dl_p_rmse = np.sqrt(((dl_p_predicted - dl_Y2_test) ** 2).mean(axis=0))
dl_p_rmsle = np.sqrt((((np.log(dl_p_predicted+1) - np.log(dl_Y2_test+1)) ** 2).mean(axis=0)))

print(dl_p_mae)
print(dl_p_mape)
print(dl_p_rmse)
print(dl_p_rmsle)

# .h5 파일 저장
painting_deeplearning_model.save('painting_DL_model.h5')

########################################################################################################################
# 분석결과 저장
dl_evaluation = {'MAE' :  [dl_m_mae[0], dl_p_mae[0]], 'MAPE' :  [dl_m_mape[0], dl_p_mape[0]], 'RMSE' :  [dl_m_rmse[0], dl_p_rmse[0]], 'RMSLE' : [dl_m_rmsle[0], dl_p_rmsle[0]]}

dl_evaluation = pd.DataFrame(dl_evaluation, index = ['making_deeplearning','painting_deeplearning'])

print(dl_evaluation)

# 분석결과 .csv 파일 저장
dl_evaluation.to_csv('spool_deeplearning_conclusion.csv', sep=',', na_rep='NaN')