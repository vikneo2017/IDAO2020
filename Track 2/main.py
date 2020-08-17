import pandas as pd
import joblib

test=pd.read_csv('test.csv')
X_test=test[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]
# Prediction
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)
predicted = loaded_model.predict(X_test)
predicted=pd.DataFrame(predicted).rename(columns={0:"x", 1:"y",2:"z",3:"Vx", 4:"Vy",5:"Vz"})
submission=pd.DataFrame(columns=['id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
submission.loc[:,'id']=test.loc[:,'id']
submission.loc[:,'x']=test.loc[:,'x_sim']
submission.loc[:,'y']=test.loc[:,'y_sim']
submission.loc[:,'z']=test.loc[:,'z_sim']
submission.loc[:,'Vx']=test.loc[:,'Vx_sim']
submission.loc[:,'Vy']=test.loc[:,'Vy_sim']
submission.loc[:,'Vz']=test.loc[:,'Vz_sim']
submission['x']=submission['x']+1.33*predicted['x']
submission['y']=submission['y']+1.33*predicted['y']
submission['z']=submission['z']+1.33*predicted['z']
submission['Vx']=submission['Vx']+1.33*predicted['Vx']
submission['Vy']=submission['Vy']+1.33*predicted['Vy']
submission['Vz']=submission['Vz']+1.33*predicted['Vz']
submission=submission[['id', 'x', 'y', 'z', 'Vx', 'Vy', 'Vz']]
submission.to_csv("submission.csv", index=False)