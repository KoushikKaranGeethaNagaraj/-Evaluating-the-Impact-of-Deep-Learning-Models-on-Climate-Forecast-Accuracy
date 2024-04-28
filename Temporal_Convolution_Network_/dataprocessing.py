import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch

##Note the csv file after index must contain Date TIme format ex: '2010-01-01 00:10:00' 
class DataProcessing:
    def __init__(self,csv_file_name,input_sequence_length,output_sequence_length,args,shuffle=True):
        self.file_name=csv_file_name
        self.input_sequence_length=input_sequence_length
        self.output_sequence_length=output_sequence_length
        self.shuffle=shuffle
        self.args=args
        self.create_df()
        self.timestamp_processing()
        self.normalise()
        self.model_data()

    
    #creates the df table reading from csv file  
    def create_df(self):
        print("Created df Table for forecasting")
        self.df = pd.read_csv(self.file_name, parse_dates=[0])
        self.df[self.df.columns[0]] = pd.to_datetime(self.df[self.df.columns[0]], format='%d.%m.%Y %H:%M:%S')
        self.df.rename(columns={self.df.columns[0]: 'timestamp'}, inplace=True)
        # Convert 'timestamp' to datetime if it isn't parsed correctly (optional)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    #converting timestamp details to useful features
    def timestamp_processing(self):
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['timestamp'].dt.hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['timestamp'].dt.hour / 24)
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['year'] = self.df['timestamp'].dt.year  # Extract year
        self.df['month'] = self.df['timestamp'].dt.month  # Extract year
        self.df['day'] = self.df['timestamp'].dt.day  # Extract year
        #splitting time stamps as its not needed while training 
        self.df_timestamps = self.df['timestamp'] 
        self.df = self.df.drop('timestamp', axis=1)  # Now df contains only the features
        self.data_np = self.df.to_numpy(dtype='float32')
        
    # Normalise the data
    def normalise(self):
        reshaped_data = self.data_np.reshape(-1, self.data_np.shape[-1])
        self.data_mean = reshaped_data.mean(axis=0)
        data_std=reshaped_data.std(axis=0)
        self.std_data = np.where(data_std > 0, data_std, 1)  # Replace 0s in std with 1 to avoid division by zero
        normalised_data = (reshaped_data - self.data_mean) / self.std_data
        # Reshape data to collapse the first three dimensions  
        self.normalised_data = normalised_data.reshape(self.data_np.shape)
        
    def create_sequences(self,normalised_data,df_timestamps):
        #creates time series sequences
        X, y, timestamps = [], [], []
        for i in range(len(normalised_data ) - self.input_sequence_length-self.output_sequence_length+1):
            X.append(normalised_data[i:(i + self.input_sequence_length)])
            y.append(normalised_data[i + (self.input_sequence_length-self.args.forecast_factor):i+(self.input_sequence_length-self.args.forecast_factor)+self.output_sequence_length])
            timestamps.append(df_timestamps.iloc[i + (self.input_sequence_length-self.args.forecast_factor):i+(self.input_sequence_length-self.args.forecast_factor)+self.output_sequence_length])
        X,Y,timestamps=np.array(X), np.array(y), np.array(timestamps)
        return X,Y,timestamps
    
        
        
    def model_data(self):
        train_test_split_index = int(len(self.normalised_data) * self.args.train_test_split)
        normalised_data_train=self.normalised_data[:train_test_split_index]
        normalised_data_test=self.normalised_data[train_test_split_index:]
        timestamps_train=self.df_timestamps[:train_test_split_index]
        timestamps_test=self.df_timestamps[train_test_split_index:]
        
        self.X_train,  self.y_train, self.timestamps_train=self.create_sequences(normalised_data_train,timestamps_train)
        self.X_test,self.y_test, self.timestamps_test =self.create_sequences(normalised_data_test,timestamps_test)
        self.inference_data,self.timestamps_data=normalised_data_test,timestamps_test
        
        if self.shuffle:
            np.random.seed(42)  # For reproducibility
            shuffled_indices = np.random.permutation(len(self.X_train))
            self.X_train = self.X_train[shuffled_indices]
            self.y_train = self.y_train[shuffled_indices]
            self.timestamps_train = self.timestamps_train[shuffled_indices]
            
    def numpy_tensor_train_test(self):
        self.X_train_tensor=torch.from_numpy(self.X_train.astype(np.float32))
        self.X_test_tensor=torch.from_numpy(self.X_test.astype(np.float32))
        self.y_train_tensor=torch.from_numpy(self.y_train.astype(np.float32))
        self.y_test_tensor=torch.from_numpy(self.y_test.astype(np.float32))
        print("Loaded Dataset... Train Dataset size: ",self.X_train_tensor.shape[0],"Test Dataset size: ",self.X_test_tensor.shape[0])
        print("Input Sequence Length :",self.X_train_tensor.shape[1])
        print("Output Sequence Length :",self.y_test.shape[1])
        print("Features :",self.X_train_tensor.shape[2])
        return self.X_train_tensor, self.X_test_tensor, self.y_train_tensor, self.y_test_tensor, self.timestamps_train, self.timestamps_test
    
    
    #################Used Later for inference#########################
    def denormalise(self,predictions):
        # bringing back to original data values
        reshaped_data = predictions.reshape(-1, predictions.shape[-1])
        denormalised_data = (reshaped_data * self.std_data)+ self.data_mean
        denormalised_data = denormalised_data.reshape(predictions.shape)
        return denormalised_data
    ###########################################
    

    

        
        
    
    