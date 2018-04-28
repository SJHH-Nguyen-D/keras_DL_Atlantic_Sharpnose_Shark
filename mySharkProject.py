"""
DESCRIPTION OF GOALS OF THE PROJECT

Given the dataset on the dietary content/stomach contents of 800 odd atlanic sharpnose sharks found
along the Coast of the Gulf of Mexico, I wanted to pursue an easy question with the numeric data.
There were a lot of NaN values which I was unsure what to make of for now. The first question I wanted
to pursue was an intuitive one and it was to use the shark measurements, along with some other categorical
features to predict the maturity state of the shark specimen

"""
#=====================================IMPORT DEPENDENCIES==============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, advanced_activations
from keras.callbacks import EarlyStopping, History
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

#=====================================LOAD YOUR DATA==============================================================
file = 'BIOLOGICAL_DATA_new.xlsx'
data = pd.ExcelFile(file)
df = data.parse(1)

#=====================================PREPROCESSING===============================================================
#dropping some of the columns that a had a lot of missing values
#replacing missing values with NaN
df_dum = pd.get_dummies(df.iloc[:,0:3], drop_first=True)
df1 = df.iloc[:,4:11]


#see how many missing values there are in the data
df1.isnull().sum()

#the output shows that shark weight and stretch total length, and total length are missing or a few of them
#intuitively, we want to impute median values for those features subsetted by shark sex and shark stage in life
imputer = Imputer(strategy='median', axis=0)

#you can only run imputer on numpy arrays, so you convert your df to numpy matrix
df1_values = df1.values
df1_impute = imputer.fit_transform(df1_values)

#print how many missing values there are after the imputation
print(np.isnan(df1_impute).sum())

#convert df1_imput back into dataframe
df1_impute_df = pd.DataFrame(df1_impute, columns=df1.columns)

#combine the two split dataframes: df1_impute and df_dum
df_comb = pd.concat([df_dum,df1_impute_df], axis=1)

#determining predictors
predictors = df_comb.values

#scale your predictors data
ss = StandardScaler()
standard = StandardScaler().fit(predictors)
predictors_scaled = standard.transform(predictors)

#determining target column
target = df.loc[:,['Maturity_State']]
target = pd.get_dummies(target).values

#=====================================DEFINING THE DEFAULT MODEL/MODEL 1=================================================
#instantiate your Model 1
model = Sequential()
model.add(Dense(150, activation='relu', input_shape=(12,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(4, activation='softmax'))

#compile the model before fitting it
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=(['accuracy']))

#fitting your model
early_stopping_monitor = EarlyStopping(patience=2)
history = History()
model.fit(predictors_scaled,
          target,
          epochs=50,
          validation_split=0.3,
          callbacks=[early_stopping_monitor, history])

#=====================================DEFINE MODEL TO LOOP THROUGH FOR LEARNING RATE===========================================
def get_new_model(x):
    model = Sequential()
    model.add(Dense(150, activation='relu', input_shape=(12,)))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return(model)

"""x is the input shape"""
#=====================================DEFINE MODEL TO LOOP THROUGH FOR LEARNING RATE===========================================    
lr_to_test = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1]

from keras.optimizers import adam

for lr in lr_to_test:
    print('\n\nTesting Model with learning rate: %f\n' %lr)
    #build a new model to test, unaffected by previous models
    model = get_new_model((12,))
    #create optimizer with the specified learning rate: my_optimizer
    my_optimizer = adam(lr=lr)
    #compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=my_optimizer,
                  metrics=(['accuracy']))
    #fit the model
    model.fit(predictors_scaled,
              target,
              epochs=50,
              validation_split=0.3,
              callbacks=[early_stopping_monitor])
    
#=====================================DEFINING SECOND THE MODEL====================================================
adams = adam(lr=0.0001)
history2 = History()

model2 = Sequential()
model2.add(Dense(364, activation='relu', input_shape=(12,)))
model2.add(Dense(364, activation='relu'))
model2.add(Dense(364, activation='relu'))
model2.add(Dense(364, activation='relu'))
model2.add(Dense(364, activation='relu'))
model2.add(Dense(4, activation='softmax'))

#compile the model before fitting it
model2.compile(loss='categorical_crossentropy',
              optimizer=adams,
              metrics=(['accuracy']))

#fitting your model
early_stopping_monitor = EarlyStopping(patience=2)
model2.fit(predictors_scaled,
          target,
          epochs=50,
          validation_split=0.3,
          callbacks=[early_stopping_monitor, history2])

#=====================================PLOTTING PERFORMANCE OF MODEL 1 AND 2 SIMULTANEOUSLY===========================================
#ploting plotting the validation loss scores of two models against epochs trained
plt.plot(history.history['val_loss'], 'r', history2.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss Score')
plt.title('Model 1 vs. Model 2 val_loss')
plt.legend(['Model 1', 'Model 2'], loc='upper left')
plt.show()
print(history2.history['val_loss'])

#ploting plotting the validation accuracy of two models against epochs trained
plt.plot(history.history['val_acc'], 'r', history2.history['val_acc'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy Score')
plt.title('Model 1 vs. Model 2 val_acc')
plt.legend(['Model 1', 'Model 2'], loc='upper left')
plt.show()
print(history2.history['val_acc'])

#=====================================PLOTTING PERFORMANCE OF MODEL 1===========================================
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model 1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history.history['acc'])
print(history.history['val_acc'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history.history['loss'])
print(history.history['val_loss'])

#=====================================PLOTTING PERFORMANCE OF MODEL 2===========================================
print(history2.history.keys())
# summarize history2 for accuracy
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('Model 2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history2.history['acc'])
print(history2.history['val_acc'])

# summarize history2 for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model 2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history2.history['loss'])
print(history2.history['val_loss'])

#=====================================SAVING THE MODEL===========================================

