import numpy as np
import pickle

loaded_model=pickle.load(open('malware_predictor.sav','rb'))

def performance_prediction(input_data):
    
    input_data=(	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	8,	8,	0,	0,
                0,	0,	1,	6,	1,	1,	6,	6,	20,	19,	0,	18,	2,	35,	28,	0,	3,	
                3,	3,	3,	0,	3,	0,	11,	5,	0,	0,	0,	0,	0,	0,	38,	0,	1,	4073)

     #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] ==0):
        return 'ransom ware'
    elif (prediction[0]==1):
        return 'good ware'
    else:
        return 'ransom ware'