# Package Load
import pandas as pd
from pandas import DataFrame
import numpy as np

########################################################################################################################
# Data 불러오기
Raw_data = pd.read_csv('./data/spooldata.csv', encoding='euc-kr')

# Column 추출
Raw_data = Raw_data[['REV날짜','BOM확정일','긴급 여부','적용L/T','BLOCK','STG','SER VICE','TAG NO','연번','SPOOL NO','대표문제점','Line No',
                     '모자재번호','설치 PLT No','최종 PLT','길이 (mm)','관통','Sch','재질','DIA','W/O일','가공일','용접일','제작일','검사일',
                     'NDE일','제작배재일','제작반출일','도장입고','도장완료','도장반출','도금반출','입고일','선별착수일','적치일',
                     '설치불출일','불출요청일','On-Deck','In-Position','설치Fit-Up','설치일','중량','수량','부재 수','Joint 수','제작협력사',
                     '후1협력사','후2협력사','물류장']]

# Column명 영어로 정리
Raw_data = Raw_data.rename(columns={'REV날짜':'D_REV','BOM확정일':'D_BOM','긴급 여부':'Emergency','적용L/T':'ApplyLeadTime','SER VICE':'Service',
                                    'TAG NO':'NO_TAG','연번':'NO_Serial','SPOOL NO':'NO_SPOOL','대표문제점':'Problem','Line No':'NO_Line',
                                    '모자재번호':'NO_Base','설치 PLT No':'NO_SetPLT','최종 PLT':'NO_PLT','길이 (mm)':'Length','관통':'Pass',
                                    '재질':'Material','W/O일':'D_WO','가공일':'D_Processing','용접일':'D_Welding','제작일':'D_Making','검사일':'D_Test',
                                    'NDE일':'D_NDE','제작배재일':'D_MakingPaking','제작반출일':'D_MakingOut','도장입고':'D_Painting1IN','도장완료':'D_Painting1END',
                                    '도장반출':'D_Painting1Out','도금반출':'D_Painting2Out','입고일':'D_Enter','선별착수일':'D_SelectStart','적치일':'D_Pile',
                                    '설치불출일':'D_NOT_Set','불출요청일':'D_NOT_Ask','On-Deck':'On_Deck','In-Position':'In_Position','설치Fit-Up':'Set_FITUP',
                                    '설치일':'D_Set','중량':'Weight','수량':'Counting','부재 수':'MemberCount','Joint 수':'JointCount','제작협력사':'Making_Co',
                                    '후1협력사':'After1_Co','후2협력사':'After2_Co','물류장':'Distribution'})

# Data Type 변경
Raw_data["Emergency"] = Raw_data["Emergency"].astype(np.object)
Raw_data["NO_Serial"] = Raw_data["NO_Serial"].astype(np.object)

Raw_data["D_REV"] = Raw_data["D_REV"].astype('datetime64[ns]')
Raw_data["D_BOM"] = Raw_data["D_BOM"].astype('datetime64[ns]')
Raw_data["D_WO"] = Raw_data["D_WO"].astype('datetime64[ns]')
Raw_data["D_Processing"] = Raw_data["D_Processing"].astype('datetime64[ns]')
Raw_data["D_Welding"] = Raw_data["D_Welding"].astype('datetime64[ns]')
Raw_data["D_Making"] = Raw_data["D_Making"].astype('datetime64[ns]')
Raw_data["D_Test"] = Raw_data["D_Test"].astype('datetime64[ns]')
Raw_data["D_NDE"] = Raw_data["D_NDE"].astype('datetime64[ns]')
Raw_data["D_MakingPaking"] = Raw_data["D_MakingPaking"].astype('datetime64[ns]')
Raw_data["D_MakingOut"] = Raw_data["D_MakingOut"].astype('datetime64[ns]')
Raw_data["D_Painting1IN"] = Raw_data["D_Painting1IN"].astype('datetime64[ns]')
Raw_data["D_Painting1END"] = Raw_data["D_Painting1END"].astype('datetime64[ns]')
Raw_data["D_Painting1Out"] = Raw_data["D_Painting1Out"].astype('datetime64[ns]')
Raw_data["D_Painting2Out"] = Raw_data["D_Painting2Out"].astype('datetime64[ns]')
Raw_data["D_Enter"] = Raw_data["D_Enter"].astype('datetime64[ns]')
Raw_data["D_SelectStart"] = Raw_data["D_SelectStart"].astype('datetime64[ns]')
Raw_data["D_Pile"] = Raw_data["D_Pile"].astype('datetime64[ns]')
Raw_data["D_NOT_Set"] = Raw_data["D_NOT_Set"].astype('datetime64[ns]')
Raw_data["D_NOT_Ask"] = Raw_data["D_NOT_Ask"].astype('datetime64[ns]')
Raw_data["On_Deck"] = Raw_data["On_Deck"].astype('datetime64[ns]')
Raw_data["In_Position"] = Raw_data["In_Position"].astype('datetime64[ns]')
Raw_data["Set_FITUP"] = Raw_data["Set_FITUP"].astype('datetime64[ns]')
Raw_data["D_Set"] = Raw_data["D_Set"].astype('datetime64[ns]')

# MakingLT 계산
MakingLT = Raw_data["D_MakingOut"]- Raw_data["D_WO"]
MakingLT = MakingLT.dt.days
MakingLT = MakingLT + 1
MakingLT = DataFrame(MakingLT)

# PaintingLT 계산
PaintingLT = Raw_data["D_Painting1Out"]- Raw_data["D_MakingOut"]
PaintingLT = PaintingLT.dt.days
PaintingLT = PaintingLT + 1
PaintingLT = DataFrame(PaintingLT)

# data frame 결합
Raw_data = pd.concat([Raw_data,MakingLT], axis = 1)
Raw_data = Raw_data.rename(columns={0:"MakingLT"})

Raw_data = pd.concat([Raw_data,PaintingLT], axis = 1)
Raw_data = Raw_data.rename(columns={0:"PaintingLT"})

Raw_data.to_csv('./data/p_spooldata.csv', sep=',', na_rep='NaN', encoding='euc-kr')

########################################################################################################################
# MakingData
MakingData = Raw_data[['DIA', 'Length', 'Weight', 'MemberCount','JointCount','Emergency','ApplyLeadTime',
                       'BLOCK','STG','Service','NO_Serial','Problem','NO_Base','NO_SetPLT','Pass','Sch',
                       'Material','Making_Co','After2_Co','Distribution','MakingLT']]

# MakingData 결측값 포함 행 삭제
MakingData = MakingData.dropna(how = "any")

MakingData.to_csv('./data/makingdata.csv', sep=',', na_rep='NaN', encoding='euc-kr')

########################################################################################################################
# PaintingData
PaintingData = Raw_data[['DIA', 'Length', 'Weight', 'MemberCount','JointCount','Emergency','ApplyLeadTime',
                       'BLOCK','STG','Service','NO_Serial','Problem','NO_Base','NO_SetPLT','Pass','Sch',
                       'Material','Making_Co','After2_Co','Distribution','PaintingLT']]

# PaintingData 결측값 포함 행 삭제
PaintingData = PaintingData.dropna(how = "any")

PaintingData.to_csv('./data/paintingdata.csv', sep=',', na_rep='NaN', encoding='euc-kr')

