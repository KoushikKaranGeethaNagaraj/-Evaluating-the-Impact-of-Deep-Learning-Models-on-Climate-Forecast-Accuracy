import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tcn_classic import ClassicTCNModel
from tcn_pytorch import TCN
from helper_functions import *
from torch.utils.data import TensorDataset, DataLoader
from dataprocessing import DataProcessing
import gc
import argparse
torch.cuda.empty_cache()
gc.collect()

class TCNRun:
    def __init__(self,args):
        if torch.cuda.is_available():
            print("CUDA is available! Using GPU.")
            self.device="cuda:0"
        else:
            print("CUDA is not available. Using CPU.")
            self.device="cpu:0"
        self.args=args
    
    def load_data(self):
        data=DataProcessing(self.args.data_file_path,self.args.input_train_sequence,self.args.output_train_sequence,args,self.args.data_shuffle)
        train_class_1=TrainEval()
        train_class_1.std_data,train_class_1.data_mean=data.std_data,data.data_mean

        X_train_tensor,X_test_tensor, y_train_tensor, y_test_tensor, timestamps_train_np, timestamps_test_np=data.numpy_tensor_train_test()
        #Loaded dataset from csv file and converted to tensor
        print(X_train_tensor.shape, X_test_tensor.shape, y_train_tensor.shape, y_test_tensor.shape, timestamps_train_np.shape, timestamps_test_np.shape)
        
        # Combining train and test for pytorch dataloader
        train_dataset=TensorDataset(X_train_tensor,y_train_tensor)
        test_dataset=TensorDataset(X_test_tensor,y_test_tensor)
        
        # Model parameters
        input_size = X_train_tensor.shape[-1]  # Number of input features
        output_size = X_train_tensor.shape[-1]  # Predicting the same number of features as the input/Forecasting features
        
        # Create the TCN model
        #Custom Made Model
        model = ClassicTCNModel(input_size, output_size,kernel_size=self.args.kernel_size,num_layers= self.args.num_channels,dilations=self.args.dilations , batch_size=self.args.train_batch_size,out_sequence=self.args.output_train_sequence,dropout=self.args.dropout)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}')
        #Pre existing Model-pytorch
        # model=TCN(
        #             input_size,
        #             self.args.num_channels,
        #             self.args.kernel_size,
        #             self.args.dilations,
        #             None,
        #             self.args.dropout,
        #             True,
        #             'weight_norm',
        #             'relu',
        #             'xavier_uniform',
        #             False,
        #             'NLC',
        #             None,
        #             'add',
        #             True,
        #             0,
        #             None,
        #             None,)
        
        model.to(self.device)
        train_loader = DataLoader(train_dataset, self.args.train_batch_size, shuffle=False)#dont change this shuffle already shuffle happens up
        test_loader = DataLoader(test_dataset, self.args.test_batch_size, shuffle=False)##dont change this shuffle already shuffle happens up
        if self.args.loss_function=="mse":criterion = nn.MSELoss()
        if self.args.loss_function=="mae" or self.args.loss_function=="hybrid_mae" :criterion = nn.L1Loss()
        if self.args.loss_function=="hybrid_mae" :criterion = nn.L1Loss(reduction="none")
        
        optimizer = torch.optim.Adam(model.parameters(), self.args.lr)

        # Start training
        if self.args.mode=="train":
            print(" model architecture ",model)
            train_class_1.train(model, train_loader, test_loader, optimizer, criterion, self.device, self.args.epochs,self.args.save_folder_name,self.args.loss_function,self.args.features)
            
        if self.args.mode=="inference":
            # test_loader = DataLoader(test_dataset, 1, shuffle=False)##dont change this shuffle already shuffle happens up
            np.set_printoptions(formatter={'float': '{:0.2f}'.format})
            model.load_state_dict(torch.load(self.args.model_path))
            prediction_total,total_target_time,total_target_data,res_save_path=train_class_1.evaluate_recursive_test(model,data.inference_data,self.device,data.timestamps_data,self.args.save_folder_name,self.args.recursive_limit,self.args.input_train_sequence,self.args.forecast_factor)
            prediction_total=data.denormalise(prediction_total)
            total_target_data=data.denormalise(total_target_data)
            print(data.df.columns)
            #getting inference losses for each features
            train_class_1.evaluate_loss_test(model, data.inference_data,self.device, self.args.features,self.args.recursive_limit,self.args.input_train_sequence,self.args.forecast_factor)
            #plotting inference graphs for each features
            plot_results(prediction_total,total_target_time,total_target_data,res_save_path,data.df.columns,self.args.features)

            
            

            

if __name__ == "__main__":
    #This is the main file set the paramters and run this file for results
    parser = argparse.ArgumentParser("tcn")
    parser.add_argument('--data_file_path', type=str, default='WeatherJena.csv')
    parser.add_argument('--model_path', type=str, default='classic_hyperparameter_mae_mod_128_128_256_256_sq_8_f3/tcn_best_model_epoch_47.pth')
    parser.add_argument('--save_folder_name', type=str, default='classic_hyperparameter_mae_mod_128_128_256_256_sq_8_f3')
    #use mode train to train and save models and use inference to test the recursive rollout.
    parser.add_argument('--mode', type=str, default='inference',help="train/inference")
    parser.add_argument('--input_train_sequence', type=int, default=8)
    parser.add_argument('--output_train_sequence', type=int, default=8)
    #This is the factor that uses n+k-----k+m approach of sequence where m is the factor(predicting future) , based on this the predictions /sequences are made and reurive rollout
    parser.add_argument('--forecast_factor', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=6)
    parser.add_argument('--dropout', type=int, default=0.25)
    #if using classic last can be anything else out features
    parser.add_argument('--num_channels', type=list, default=[128,128,256,256])
    parser.add_argument('--dilations', type=list, default=[1,2,4,4])
    parser.add_argument('--loss_function', type=str, default="mse",help="mse/mae")
    #for classical_tcn use test and train batch size to be same or you'll get error
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--recursive_limit', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data_shuffle', type=bool, default=True)
    parser.add_argument('--train_test_split', type=float, default=0.8)#train percentage
    parser.add_argument('--features', type=list, default=[0,1,4,10,14,20])#MODEL OUTPUT/INPUT  FEATURES TO PLOT DURING INFERENCE only even pair of features

    args = parser.parse_args()
    tcn_1=TCNRun(args)
    tcn_1.load_data()
    

