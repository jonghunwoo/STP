# 해당 파일은 .h5 파일로 저장된 deeplearning모델을 불러와서 별도의 학습 과정 없이 사용가능한지 테스트를 수행
# Package Load
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

########################################### MakingLT ###################################################################
# Data 불러오기
dl_MakingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_makingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del dl_MakingLT['Unnamed: 0']

# Data Type 변경
dl_MakingLT["Emergency"] = dl_MakingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
dl_m_Emergency_one_hot_encoded = pd.get_dummies(dl_MakingLT.Emergency)
dl_m_ApplyLeadTime_one_hot_encoded = pd.get_dummies(dl_MakingLT.ApplyLeadTime)
dl_m_STG_one_hot_encoded = pd.get_dummies(dl_MakingLT.STG)
dl_m_Service_one_hot_encoded = pd.get_dummies(dl_MakingLT.Service)
dl_m_Pass_one_hot_encoded = pd.get_dummies(dl_MakingLT.Pass)
dl_m_Sch_one_hot_encoded = pd.get_dummies(dl_MakingLT.Sch)
dl_m_Material_one_hot_encoded = pd.get_dummies(dl_MakingLT.Material)
dl_m_Making_Co_one_hot_encoded = pd.get_dummies(dl_MakingLT.Making_Co)

# Input, Output data 분류
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

# 학습모델 구축을 위해 data 형식을 Vector로 변환
dl_X1 = dl_m_Inputdata.values
dl_Y1 = dl_m_Outputdata.values

# 데이터 표준화 적용 (deep learning의 경우, input 데이터 표준화 적용 시 오차율이 감소함을 확인)
scaler = MinMaxScaler(feature_range = (0,1))
dl_X1_scale = scaler.fit_transform(dl_X1)

# Training Data, Test Data 분리
dl_X1_train, dl_X1_test, dl_Y1_train, dl_Y1_test = train_test_split(dl_X1_scale, dl_Y1, test_size = 0.33, random_state = 42)

########################################### h5 file ###################################################################
# .h5파일로 저장된 deeplearning 모델을 불러와서 별도의 모델 학습과정 없이 사용
from keras.models import load_model
dl_m_model = load_model('making_DL_model.h5')

dl_m_predicted = dl_m_model.predict(dl_X1_test)

dl_m_mae = abs(dl_m_predicted - dl_Y1_test).mean(axis=0)
dl_m_mape = (np.abs((dl_m_predicted - dl_Y1_test) / dl_Y1_test).mean(axis=0))
dl_m_rmse = np.sqrt(((dl_m_predicted - dl_Y1_test) ** 2).mean(axis=0))
dl_m_rmsle = np.sqrt((((np.log(dl_m_predicted+1) - np.log(dl_Y1_test+1)) ** 2).mean(axis=0)))

print(dl_m_mae)
print(dl_m_mape)
print(dl_m_rmse)
print(dl_m_rmsle)

########################################### PaintingLT #################################################################
# Data 불러오기
dl_PaintingLT = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/p_paintingdata.csv',engine = 'python')

# 첫번째 Column 삭제
del dl_PaintingLT['Unnamed: 0']

# Data Type 변경
dl_PaintingLT["Emergency"] = dl_PaintingLT["Emergency"].astype(np.object)

# One-hot encoding (범주형 데이터를 컴퓨터가 인식할 수 있는 형태 즉, 문자를 숫자로 변환하는 단계)
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
dl_Y2 = dl_p_Outputdata.values

# 데이터 표준화 적용 (deep learning의 경우, input 데이터 표준화 적용 시 오차율이 감소함을 확인)
dl_X2_scale = scaler.fit_transform(dl_X2)

# Training Data, Test Data 분리
dl_X2_train, dl_X2_test, dl_Y2_train, dl_Y2_test = train_test_split(dl_X2_scale, dl_Y2, test_size = 0.33, random_state = 42)

########################################### h5 file ###################################################################
# .h5파일로 저장된 deeplearning 모델을 불러와서 별도의 모델 학습과정 없이 사용
from keras.models import load_model
dl_p_model = load_model('painting_DL_model.h5')

dl_p_predicted = dl_p_model.predict(dl_X2_test)

dl_p_mae = abs(dl_p_predicted - dl_Y2_test).mean(axis=0)
dl_p_mape = (np.abs((dl_p_predicted - dl_Y2_test) / dl_Y2_test).mean(axis=0))
dl_p_rmse = np.sqrt(((dl_p_predicted - dl_Y2_test) ** 2).mean(axis=0))
dl_p_rmsle = np.sqrt((((np.log(dl_p_predicted+1) - np.log(dl_Y2_test+1)) ** 2).mean(axis=0)))

print(dl_p_mae)
print(dl_p_mape)
print(dl_p_rmse)
print(dl_p_rmsle)