import pandas as pd
import numpy as np
from keras import models
from keras import layers
from numpy import savetxt
from keras import regularizers
from keras import optimizers
''' Zakres oznaczony #---- został wygenerowany z pomocą skryptu pythona a nie ręcznie :)  '''
#---------
MSZoning = { 'nan':80,'A': 1,'C': 2,'FV': 3,'I': 4,'RH': 5,'RL': 6,'RP': 7,'RM': 8,}
Street = { 'nan':80,'Grvl': 1,'Pave': 2,}
Alley = { 'nan':80,'Grvl': 1,'Pave': 2,'NA': 3,}
LotShape = { 'nan':80,'Reg': 1,'IR1': 2,'IR2': 3,'IR3': 4,}
LandContour = { 'nan':80,'Lvl': 1,'Bnk': 2,'HLS': 3,'Low': 4,}
Utilities = { 'nan':80,'AllPub': 1,'NoSewr': 2,'NoSeWa': 3,'ELO': 4,}
LotConfig = { 'nan':80,'Inside': 1,'Corner': 2,'CulDSac': 3,'FR2': 4,'FR3': 5,}
LandSlope = { 'nan':80,'Gtl': 1,'Mod': 2,'Sev': 3,}
Neighborhood = { 'nan':80,'Blmngtn': 1,'Blueste': 2,'BrDale': 3,'BrkSide': 4,'ClearCr': 5,'CollgCr': 6,'Crawfor': 7,'Edwards': 8,'Gilbert': 9,'IDOTRR': 10,'MeadowV': 11,'Mitchel': 12,'Names': 13,'NoRidge': 14,'NPkVill': 15,'NridgHt': 16,'NWAmes': 17,'OldTown': 18,'SWISU': 19,'Sawyer': 20,'SawyerW': 21,'Somerst': 22,'StoneBr': 23,'Timber': 24,'Veenker': 25,}
Condition1 = { 'nan':80,'Artery': 1,'Feedr': 2,'Norm': 3,'RRNn': 4,'RRAn': 5,'PosN': 6,'PosA': 7,'RRNe': 8,'RRAe': 9,}
Condition2 = { 'nan':80,'Artery': 1,'Feedr': 2,'Norm': 3,'RRNn': 4,'RRAn': 5,'PosN': 6,'PosA': 7,'RRNe': 8,'RRAe': 9,}
BldgType = { 'nan':80,'1Fam': 1,'2FmCon': 2,'Duplx': 3,'TwnhsE': 4,'TwnhsI': 5,}
HouseStyle = { 'nan':80,'1Story': 1,'1.5Fin': 2,'1.5Unf': 3,'2Story': 4,'2.5Fin': 5,'2.5Unf': 6,'SFoyer': 7,'SLvl': 8,}
RoofStyle = { 'nan':80,'Flat': 1,'Gable': 2,'Gambrel': 3,'Hip': 4,'Mansard': 5,'Shed': 6,}
RoofMatl = { 'nan':80,'ClyTile': 1,'CompShg': 2,'Membran': 3,'Metal': 4,'Roll': 5,'Tar&Grv': 6,'WdShake': 7,'WdShngl': 8,}
Exterior1st = { 'nan':80,'AsbShng': 1,'AsphShn': 2,'BrkComm': 3,'BrkFace': 4,'CBlock': 5,'CemntBd': 6,'HdBoard': 7,'ImStucc': 8,'MetalSd': 9,'Other': 10,'Plywood': 11,'PreCast': 12,'Stone': 13,'Stucco': 14,'VinylSd': 15,'Wd': 16,'WdShing': 17,}
Exterior2nd = { 'nan':80,'AsbShng': 1,'AsphShn': 2,'BrkComm': 3,'BrkFace': 4,'CBlock': 5,'CemntBd': 6,'HdBoard': 7,'ImStucc': 8,'MetalSd': 9,'Other': 10,'Plywood': 11,'PreCast': 12,'Stone': 13,'Stucco': 14,'VinylSd': 15,'Wd': 16,'WdShing': 17,}
MasVnrType = { 'nan':80,'BrkCmn': 1,'BrkFace': 2,'CBlock': 3,'None': 4,'Stone': 5,}
ExterQual = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,}
ExterCond = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,}
Foundation = { 'nan':80,'BrkTil': 1,'CBlock': 2,'PConc': 3,'Slab': 4,'Stone': 5,'Wood': 6,}
BsmtQual = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,'NA': 6,}
BsmtCond = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,'NA': 6,}
BsmtExposure = { 'nan':80,'Gd': 1,'Av': 2,'Mn': 3,'No': 4,'NA': 5,}
BsmtFinType1 = { 'nan':80,'GLQ': 1,'ALQ': 2,'BLQ': 3,'Rec': 4,'LwQ': 5,'Unf': 6,'NA': 7,}
BsmtFinType2 = { 'nan':80,'GLQ': 1,'ALQ': 2,'BLQ': 3,'Rec': 4,'LwQ': 5,'Unf': 6,'NA': 7,}
Heating = { 'nan':80,'Floor': 1,'GasA': 2,'GasW': 3,'Grav': 4,'OthW': 5,'Wall': 6,}
HeatingQC = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,}
CentralAir = { 'nan':80,'N': 1,'Y': 2,}
Electrical = { 'nan':80,'SBrkr': 1,'FuseA': 2,'FuseF': 3,'FuseP': 4,'Mix': 5,}
KitchenQual = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,}
Functional = { 'nan':80,'Typ': 1,'Min1': 2,'Min2': 3,'Mod': 4,'Maj1': 5,'Maj2': 6,'Sev': 7,'Sal': 8,}
FireplaceQu = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,'NA': 6,}
GarageType = { 'nan':80,'2Types': 1,'Attchd': 2,'Basment': 3,'BuiltIn': 4,'CarPort': 5,'Detchd': 6,'NA': 7,}
GarageFinish = { 'nan':80,'Fin': 1,'RFn': 2,'Unf': 3,'NA': 4,}
GarageQual = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,'NA': 6,}
GarageCond = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'Po': 5,'NA': 6,}
PavedDrive = { 'nan':80,'Y': 1,'P': 2,'N': 3,}
PoolQC = { 'nan':80,'Ex': 1,'Gd': 2,'TA': 3,'Fa': 4,'NA': 5,}
Fence = { 'nan':80,'GdPrv': 1,'MnPrv': 2,'GdWo': 3,'MnWw': 4,'NA': 5,}
MiscFeature = { 'nan':80,'Elev': 1,'Gar2': 2,'Othr': 3,'Shed': 4,'TenC': 5,'NA': 6,}
SaleType = { 'nan':80,'WD': 1,'CWD': 2,'VWD': 3,'New': 4,'COD': 5,'Con': 6,'ConLw': 7,'ConLI': 8,'ConLD': 9,'Oth': 10,}
SaleCondition = { 'nan':80,'Normal': 1,'Abnorml': 2,'AdjLand': 3,'Alloca': 4,'Family': 5,'Partial': 6,}



DataFrame = pd.read_csv('D:\Python\Programy\Tutorial\Kaggle\House Prices\\train.csv',keep_default_na=False)
DataFrame['MSZoning'] = DataFrame['MSZoning'].map(MSZoning)
DataFrame['Street']  = DataFrame['Street'].map(Street)
DataFrame['Alley']  = DataFrame['Alley'].map(Alley)
DataFrame['LotShape']  = DataFrame['LotShape'].map(LotShape)
DataFrame['LandContour']  = DataFrame['LandContour'].map(LandContour)
DataFrame['Utilities']  = DataFrame['Utilities'].map(Utilities)
DataFrame['LotConfig']  = DataFrame['LotConfig'].map(LotConfig)
DataFrame['LandSlope']  = DataFrame['LandSlope'].map(LandSlope)
DataFrame['Neighborhood']  = DataFrame['Neighborhood'].map(Neighborhood)
DataFrame['Condition1']  = DataFrame['Condition1'].map(Condition1)
DataFrame['Condition2']  = DataFrame['Condition2'].map(Condition2)
DataFrame['BldgType']  = DataFrame['BldgType'].map(BldgType)
DataFrame['HouseStyle']  = DataFrame['HouseStyle'].map(HouseStyle)
DataFrame['RoofStyle']  = DataFrame['RoofStyle'].map(RoofStyle)
DataFrame['RoofMatl']  = DataFrame['RoofMatl'].map(RoofMatl)
DataFrame['Exterior1st']  = DataFrame['Exterior1st'].map(Exterior1st)
DataFrame['Exterior2nd']  = DataFrame['Exterior2nd'].map(Exterior2nd)
DataFrame['MasVnrType']  = DataFrame['MasVnrType'].map(MasVnrType)
DataFrame['ExterQual']  = DataFrame['ExterQual'].map(ExterQual)
DataFrame['ExterCond']  = DataFrame['ExterCond'].map(ExterCond)
DataFrame['Foundation']  = DataFrame['Foundation'].map(Foundation)
DataFrame['BsmtQual']  = DataFrame['BsmtQual'].map(BsmtQual)
DataFrame['BsmtCond']  = DataFrame['BsmtCond'].map(BsmtCond)
DataFrame['BsmtExposure']  = DataFrame['BsmtExposure'].map(BsmtExposure)
DataFrame['BsmtFinType1']  = DataFrame['BsmtFinType1'].map(BsmtFinType1)
DataFrame['BsmtFinType2']  = DataFrame['BsmtFinType2'].map(BsmtFinType2)
DataFrame['Heating']  = DataFrame['Heating'].map(Heating)
DataFrame['HeatingQC']  = DataFrame['HeatingQC'].map(HeatingQC)
DataFrame['CentralAir']  = DataFrame['CentralAir'].map(CentralAir)
DataFrame['Electrical']  = DataFrame['Electrical'].map(Electrical)
DataFrame['KitchenQual']  = DataFrame['KitchenQual'].map(KitchenQual)
DataFrame['Functional']  = DataFrame['Functional'].map(Functional)
DataFrame['FireplaceQu']  = DataFrame['FireplaceQu'].map(FireplaceQu)
DataFrame['GarageType']  = DataFrame['GarageType'].map(GarageType)
DataFrame['GarageFinish']  = DataFrame['GarageFinish'].map(GarageFinish)
DataFrame['GarageQual']  = DataFrame['GarageQual'].map(GarageQual)
DataFrame['GarageCond']  = DataFrame['GarageCond'].map(GarageCond)
DataFrame['PavedDrive']  = DataFrame['PavedDrive'].map(PavedDrive)
DataFrame['PoolQC']  = DataFrame['PoolQC'].map(PoolQC)
DataFrame['Fence']  = DataFrame['Fence'].map(Fence)
DataFrame['MiscFeature']  = DataFrame['MiscFeature'].map(MiscFeature)
DataFrame['SaleType']  = DataFrame['SaleType'].map(SaleType)
DataFrame['SaleCondition']  = DataFrame['SaleCondition'].map(SaleCondition)
#---------

DataFrame = DataFrame.replace('NA',0)
DataFrame = DataFrame.fillna(80)
lables = ['PoolArea','LotConfig','Condition1','MiscFeature','LandContour','HouseStyle','Condition2','LandSlope','MoSold','3SsnPorch',
 'Street','MasVnrType', 'BsmtFinType2','BsmtFinSF2', 'Utilities', 'BsmtHalfBath', 'ExterCond','MiscVal' ,'LowQualFinSF','YrSold','Electrical',
 'Exterior2nd', 'OverallCond','MSSubClass']
DataFrame = DataFrame.drop(labels=lables,axis=1)

'''DataFrame=DataFrame.sample(frac=1).reset_index(drop=True)
pd.set_option("display.max_rows", None, "display.max_columns", None)
lel = DataFrame.corr().sort_values(by='SalePrice',ascending=False)
print(lel['SalePrice'])'''

Data = np.array(DataFrame.iloc[:,1:len(DataFrame.columns)-1],dtype=float)
DataTarget= np.array(DataFrame['SalePrice'],dtype=float)

mean = Data.mean(axis=0)
std=Data.std(axis=0)

Data -= mean
Data /= std


Save = pd.DataFrame(np.stack((mean,std),axis=0))
Save.to_csv('np.txt',index=False)

model = models.Sequential()
model.add(layers.Dense(40,activation='relu',input_shape=(55,), kernel_regularizer=regularizers.l2(0.1) ))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(20,activation='relu', kernel_regularizer=regularizers.l1(0.1)))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.Adam(learning_rate=0.008), loss='mse',metrics=['mae'])

k=6 #6
num_val_samples = len(Data) // k
all_mae_histories = []
all_mae_histories2 = []
num_epochs = 10
for i in range(k):
    val_data = Data[ i * num_val_samples: (i + 1) * num_val_samples ]
    val_targets = DataTarget [ i * num_val_samples: (i + 1) * num_val_samples ]

    partial_train_data = np.concatenate( [ Data [ :i * num_val_samples ],  Data [ (i + 1) * num_val_samples: ] ], axis=0)
    partial_train_targets = np.concatenate( [ DataTarget [ :i * num_val_samples ],DataTarget [ (i + 1) * num_val_samples: ] ],axis=0)

    history = model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,validation_data=(val_data,val_targets),batch_size=128)
    mae_history = history.history[ 'val_mae' ]
    mae_history2 = history.history[ 'mae' ]
    all_mae_histories.append(mae_history)
    all_mae_histories2.append(mae_history2)

    '''UsedValues = []
    val_data= []
    val_targets =[]
    for i in range(146):
        import random
        loop = True
        while loop:
           index = random.randint(0,1459)
           if not(index in UsedValues) :
               UsedValues.append(index)
               val_data.append(Data[index])
               val_targets.append(DataTarget[index])
               loop=False

    val_data = np.array(val_data,dtype=float)
    val_targets =np.array(val_targets,dtype=float)

    partial_train_data = np.delete(Data,UsedValues,axis=0)
    partial_train_targets = np.delete(DataTarget,UsedValues,axis=0)'''


model.save('KaggleModel1.h5')


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
average_mae_history2 = [np.mean([x[i] for x in all_mae_histories2]) for i in range(num_epochs)]
import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.9):
     smoothed_points = []
     for point in points:
         if smoothed_points:
             previous = smoothed_points[-1]
             smoothed_points.append(previous * factor + point * (1 - factor))
         else:
             smoothed_points.append(point)
     return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[100:])
smooth_mae_history2 = smooth_curve(average_mae_history2[100:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.plot(range(1, len(smooth_mae_history2) + 1), smooth_mae_history2,'bo')
plt.xlabel('Liczba epok')
plt.ylabel('Sredni blad bezwzgledny')
plt.show()
