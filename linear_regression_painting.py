import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model

PaintingLT = pd.read_csv('./data/PaintingData2.csv',engine = 'python')

# 첫번째 Column 삭제
del PaintingLT['Unnamed: 0']

# Data Type 변경
PaintingLT["Emergency"] = PaintingLT["Emergency"].astype(np.object)

# 더미변수 생성(One-hot encoding)
# 모든 범주형 변수에 적용
# Dummy : 범주형 변수를 연속형 변수로 변환하기 위해 사용하며, 해당하는 칸의 정보를 1, 나머지를 0으로 표시한다.
# concat : cbind (axis =1), rbind (axis = 2)

Emergency_one_hot_encoded = pd.get_dummies(PaintingLT.Emergency)
Emergency_with_one_hot = pd.concat([PaintingLT.Emergency, Emergency_one_hot_encoded], axis = 1)

ApplyLeadTime_one_hot_encoded = pd.get_dummies(PaintingLT.ApplyLeadTime)
ApplyLeadTime_with_one_hot = pd.concat([PaintingLT.ApplyLeadTime, ApplyLeadTime_one_hot_encoded], axis = 1)

STG_one_hot_encoded = pd.get_dummies(PaintingLT.STG)
STG_with_one_hot = pd.concat([PaintingLT.STG, STG_one_hot_encoded], axis = 1)

Service_one_hot_encoded = pd.get_dummies(PaintingLT.Service)
Service_with_one_hot = pd.concat([PaintingLT.Service, Service_one_hot_encoded], axis = 1)

Pass_one_hot_encoded = pd.get_dummies(PaintingLT.Pass)
Pass_with_one_hot = pd.concat([PaintingLT.Pass, Pass_one_hot_encoded], axis = 1)

Sch_one_hot_encoded = pd.get_dummies(PaintingLT.Sch)
Sch_with_one_hot = pd.concat([PaintingLT.Sch, Sch_one_hot_encoded], axis = 1)

Material_one_hot_encoded = pd.get_dummies(PaintingLT.Material)
Material_with_one_hot = pd.concat([PaintingLT.Material, Material_one_hot_encoded], axis = 1)

Making_Co_one_hot_encoded = pd.get_dummies(PaintingLT.Making_Co)
Making_Co_with_one_hot = pd.concat([PaintingLT.Making_Co, Making_Co_one_hot_encoded], axis = 1)

After2_Co_one_hot_encoded = pd.get_dummies(PaintingLT.After2_Co)
After2_Co_with_one_hot = pd.concat([PaintingLT.After2_Co, After2_Co_one_hot_encoded], axis = 1)

Inputdata = pd.concat((PaintingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , Emergency_one_hot_encoded
                      , ApplyLeadTime_one_hot_encoded
                      , STG_one_hot_encoded
                      , Service_one_hot_encoded
                      , Pass_one_hot_encoded
                      , Sch_one_hot_encoded
                      , Material_one_hot_encoded
                      , Making_Co_one_hot_encoded
                      , After2_Co_one_hot_encoded)
                      , axis=1)

Outputdata = PaintingLT[['PaintingLT']]

# Vector 변환 (Deeplearning에서는 Vector값이 사용)
# X, Y는 RawData

X = Inputdata.values
Y = Outputdata.values

# Training Data, Test Data 분리
# Test Data 비율 : 0.33, random_state : 난수 발생을 위한 Seed 인자 값 (대부분 42로 사용)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

# 선형회귀분석 모델 구축
# 모델 구성하기
painting_regression_model1 = linear_model.LinearRegression()
painting_regression_model1.fit(X_train, Y_train)
predicted1 = painting_regression_model1.predict(X_test)

mae = abs(predicted1 - Y_test).mean(axis=0)
mape = (np.abs((predicted1 - Y_test) / Y_test).mean(axis=0))
rmse = np.sqrt(((predicted1 - Y_test) ** 2).mean(axis=0))

# 분석결과 저장
evaluation = {'MAE' :  [mae[0]],
              'MAPE' :  [mape[0],],
              'RMSE' :  [rmse[0],]}

evaluation = pd.DataFrame(evaluation, index = ['case1'])
print(evaluation)

evaluation.to_csv('PaintingLT_LinearRegression_conclusion.csv', sep=',', na_rep='NaN')
