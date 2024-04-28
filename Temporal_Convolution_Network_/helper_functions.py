import torch
import numpy as np
from tqdm import tqdm
import os
import torch.nn as nn
import matplotlib.pyplot as plt


class TrainEval:
    def __init__(self):
        pass

    def train_epoch(self,model, data_loader, optimizer, criterion, device,loss_function,features):
        model.train()  # Set the model to training mode
        total_norm_mse_mae_loss=0.0
        total_loss_mse = 0.0
        total_r2_score_loss= 0.0
        total_smpae_metric_loss = 0.0
        total_mape_metric_loss = 0.0
        # Wrap your data_loader with tqdm for a progress bar
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Training', leave=False)
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)  # Move data to the appropriate device
            optimizer.zero_grad()  # Zero the gradients before running the backward pass
            output = model(data)  # Forward pass
            # print(output.shape)
            #mse /mae
            mse_mae_loss = criterion(output, target)
            #hybrid_mae
            
            if loss_function=="hybrid_mae":
                weighted_parameters=torch.full((output.shape[-1], 1), 0.01, dtype=torch.float32, device='cuda')
                weighted_parameters[features]=0.137
                mse_mae_loss = mse_mae_loss.mean(dim=(0, 1))
                mse_mae_loss=mse_mae_loss*weighted_parameters
                mse_mae_loss=torch.sum(mse_mae_loss)
                
            
            # Compute loss
            target_np=self.denormalise(tensor_numpy(target))
            output_np=self.denormalise(tensor_numpy(output))

            mse_score= mse_metric(output_np, target_np)   
            r2_score_loss=r2_score_metric(output_np, target_np)
            smpae_metric_loss=smpae_metric(output_np,target_np)
            mape_metric_loss=mape_metric(output_np,target_np)

            mse_mae_loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            total_norm_mse_mae_loss+=mse_mae_loss.item()
            total_loss_mse += mse_score
            total_r2_score_loss+=r2_score_loss
            total_smpae_metric_loss += smpae_metric_loss
            total_mape_metric_loss += mape_metric_loss
            # Update the progress bar with the latest loss.
            progress_bar.set_postfix({'loss': mse_mae_loss.item()})

        total_loss=[total_norm_mse_mae_loss/len(data_loader),total_loss_mse/ len(data_loader) ,total_r2_score_loss/ len(data_loader) ,total_smpae_metric_loss/ len(data_loader) ,total_mape_metric_loss/ len(data_loader) ]
        return total_loss,mse_mae_loss.item()   # Return average loss

    def evaluate_train(self,model, data_loader, criterion, device,loss_function,features):
        print("Evaluating...")
        model.eval()
        # The code `total_norm_mse_mae_loss` is defining a variable name in Python. It seems to be a
        # variable name that might be used to store the total normalized mean squared error and mean
        # absolute error loss in a machine learning context.
        # The code `total_norm_mse_mae_loss` is defining a variable name in Python. It does not
        # contain any actual code or functionality, it is just a variable name.
        total_norm_mse_mae_loss=0.0
        total_loss_mse = 0.0
        total_r2_score_loss= 0.0
        total_smpae_metric_loss = 0.0
        total_mape_metric_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                mse_mae_loss = criterion(output, target)
                if loss_function=="hybrid_mae":
                    weighted_parameters=torch.full((output.shape[-1], 1), 0.01, dtype=torch.float32, device='cuda')
                    weighted_parameters[features]=0.137
                    mse_mae_loss = mse_mae_loss.mean(dim=(0, 1))
                    mse_mae_loss=mse_mae_loss*weighted_parameters
                    mse_mae_loss=torch.sum(mse_mae_loss)
                target_np=self.denormalise(tensor_numpy(target))
                output_np=self.denormalise(tensor_numpy(output))
                mse_score= mse_metric(output_np, target_np) 
                r2_score_loss=r2_score_metric(output_np, target_np)
                smpae_metric_loss=smpae_metric(output_np,target_np)
                mape_metric_loss=mape_metric(output_np,target_np)
                total_norm_mse_mae_loss+=mse_mae_loss.item()
                total_loss_mse += mse_score
                total_r2_score_loss+=r2_score_loss
                total_smpae_metric_loss += smpae_metric_loss
                total_mape_metric_loss += mape_metric_loss
        total_loss=[total_norm_mse_mae_loss/len(data_loader),total_loss_mse/ len(data_loader) ,total_r2_score_loss/ len(data_loader) ,total_smpae_metric_loss/ len(data_loader) ,total_mape_metric_loss/ len(data_loader) ]
        return total_loss ,mse_mae_loss.item()

    def train(self,model, train_loader, test_loader, optimizer, criterion, device, epochs,save_folder_name,loss_function,features):
        # print(torch.cuda.memory_summary())
        os.makedirs(save_folder_name, exist_ok=True)
        loss_save_path=os.path.join(save_folder_name, 'loss')
        os.makedirs(loss_save_path, exist_ok=True)
        norm_mse_mae_tr_loss,norm_mse_mae_ts_loss=[],[]
        mse_tr_loss,mse_ts_loss=[],[]
        r2_tr_loss,r2_ts_loss=[],[]
        smpae_tr_loss,smape_ts_loss=[],[]
        mape_tr_loss,mape_ts_loss=[],[]

        prev_test_loss=np.inf
        for epoch in range(epochs):
            train_losses_norm,mse_mae_train_ls = self.train_epoch(model, train_loader, optimizer, criterion, device,loss_function,features)
            test_losses_norm, mse_mae_test_ls = self.evaluate_train(model, test_loader, criterion, device,loss_function,features)
            print(f'Epoch {epoch+1}, Train Loss: {mse_mae_train_ls:.4f}, Test Loss: {mse_mae_test_ls:.4f}')
            md_save_path = os.path.join(save_folder_name, f'tcn_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), md_save_path)
            if mse_mae_test_ls<prev_test_loss:
                prev_test_loss=mse_mae_test_ls
                save_path = os.path.join(save_folder_name, f'tcn_best_model_epoch_{epoch}.pth')
                torch.save(model.state_dict(), save_path)
            
            norm_mse_mae_tr_loss.append(train_losses_norm[0])
            mse_tr_loss.append(train_losses_norm[1])
            r2_tr_loss.append(train_losses_norm[2])
            smpae_tr_loss.append(train_losses_norm[3])
            mape_tr_loss.append(train_losses_norm[4])
            
            norm_mse_mae_ts_loss.append(test_losses_norm[0])
            mse_ts_loss.append(test_losses_norm[1])
            r2_ts_loss.append(test_losses_norm[2])
            smape_ts_loss.append(test_losses_norm[3])
            mape_ts_loss.append(test_losses_norm[4])

        mse_tr_loss,r2_tr_loss,smpae_tr_loss,mape_tr_loss=np.array(mse_tr_loss),np.array(r2_tr_loss),np.array(smpae_tr_loss),np.array(mape_tr_loss)
        mse_ts_loss,r2_ts_loss,smape_ts_loss,mape_ts_loss=np.array(mse_ts_loss),np.array(r2_ts_loss),np.array(smape_ts_loss),np.array(mape_ts_loss)
        loss_save_path=os.path.join(loss_save_path, 'train_test_losses.npz')
        plot_save_path=os.path.join(save_folder_name, 'losses_plot.png')

        plot_loss_epoch([norm_mse_mae_tr_loss,mse_tr_loss,r2_tr_loss,smpae_tr_loss,mape_tr_loss],[norm_mse_mae_ts_loss,mse_ts_loss,r2_ts_loss,smape_ts_loss,mape_ts_loss],epochs,["Normalised MSE/MAE","MSE","R2 Score","smape","mape"],plot_save_path)
        
        np.savez(loss_save_path, array1=norm_mse_mae_tr_loss,array2=mse_tr_loss,array3=r2_tr_loss,array4=smpae_tr_loss,array5=mape_tr_loss, array6=norm_mse_mae_ts_loss,array7=mse_ts_loss,array8=r2_ts_loss,array9=smape_ts_loss,array10=mape_ts_loss) 


    ###########If mode ==inference###########################
    #####recursive rollout#####################
    def evaluate_recursive_test(self,model, test_data_set, device,timestamps_test_np,save_folder_name,recursive_predictions=100,sequence_length=10,forecast_factor=0):
        #This fucntion returns the first 
        print("Evaluating...Inference")
        model.eval()
        print("Total hours predicting: ",recursive_predictions/6)
        forecast_len=sequence_length-forecast_factor
        recursive_predictions=recursive_predictions+forecast_len

        total_target_data=test_data_set[forecast_len:recursive_predictions+forecast_len]
        total_target_time=timestamps_test_np[:recursive_predictions]
        
        
        prediction_begin=torch.tensor(test_data_set[:sequence_length,:])
        prediction_total=torch.zeros(forecast_len,test_data_set.shape[1])
        
        with torch.no_grad():
            while(prediction_total.shape[0]<=total_target_data.shape[0]):
                prediction_begin = prediction_begin.unsqueeze(0).to(device)
                # print(prediction_begin[:,:,0])
                prediction = model(prediction_begin)
                # print(prediction[:,:,0])
                # print(total_target_data[0,0])
                prediction=prediction.detach().cpu().squeeze(0)
                prediction_begin=prediction
                prediction_total = torch.cat([prediction_total,prediction[forecast_len:,:]],dim=0)  
            prediction_total=prediction_total[:recursive_predictions].numpy()
            
        prediction_total=prediction_total[forecast_len:]
        total_target_time=total_target_time[forecast_len:]
        total_target_data=total_target_data[forecast_len:]
        print("recursive prediction",prediction_total.shape)
        print("target_time",total_target_time.shape)
        print("target data",total_target_data.shape)
        res_save_path=os.path.join(save_folder_name, 'ressults.png')
        return prediction_total,total_target_time,total_target_data,res_save_path
    
    def evaluate_loss_test(self,model, test_data_set, device,features,recursive_predictions=100,sequence_length=10,forecast_factor=0):
        #This fucntion returns the first 
        print("Evaluating...Loss_Inference")
        model.eval()
        
        Overall_pred_data=[]
        Overall_target_data=[]
        print("Total hours predicting: ",recursive_predictions/6)
        forecast_len=sequence_length-forecast_factor
        recursive_predictions=recursive_predictions+forecast_len
        
        
        for seq in range(0,(test_data_set.shape[0]-500),recursive_predictions):
            
            total_target_data=test_data_set[forecast_len+seq:recursive_predictions+forecast_len+seq] 
            prediction_begin=torch.tensor(test_data_set[seq:sequence_length+seq,:])
            
            prediction_total=torch.zeros(forecast_len,test_data_set.shape[1])
            prediction=None
            with torch.no_grad():
                while(prediction_total.shape[0]<=total_target_data.shape[0]):
                    prediction_begin = prediction_begin.unsqueeze(0).to(device)
                    
                    prediction = model(prediction_begin)
                    
                    prediction=prediction.detach().cpu().squeeze(0)
                    prediction_begin=prediction
                    prediction_total = torch.cat([prediction_total,prediction[forecast_len:,:]],dim=0)  
                prediction_total=prediction_total[:recursive_predictions].numpy()
                
            prediction_total=prediction_total[forecast_len:]
            total_target_data=total_target_data[forecast_len:]
            Overall_pred_data.append(prediction_total)
            Overall_target_data.append(total_target_data)
            
            
        Overall_pred_data,Overall_target_data=np.array(Overall_pred_data),np.array(Overall_target_data)
        
        print(Overall_pred_data.shape,Overall_target_data.shape)        
        mse_infer=mse_metric(Overall_target_data[:,:,features],Overall_pred_data[:,:,features])
        mae_infer=mae_metric(Overall_target_data[:,:,features],Overall_pred_data[:,:,features])
        print("mse",mse_infer,"mae",mae_infer)
            

    
    def denormalise(self,predictions):
        # bringing back to original data values
        reshaped_data = predictions.reshape(-1, predictions.shape[-1])
        denormalised_data = (reshaped_data * self.std_data)+ self.data_mean
        denormalised_data = denormalised_data.reshape(predictions.shape)
        return denormalised_data
        
############can be used outside with any input ##########
def numpy_tensor(input_np,device):
    out_tensor=torch.from_numpy(input_np.astype(np.float32)).to(device)
    return out_tensor 

def tensor_numpy(input_tensor):
    out_np = input_tensor.detach().cpu().numpy()
    return out_np



############## metrics ##############
#onnly numpy arrays
def mape_metric(pred,target):
    mape = np.mean(np.abs((((target+1e-8)  - pred)) / (target+1e-8 )) * 100) 
    return mape

def smpae_metric(pred,target):
    numerator = np.abs((pred+1e-8) - (target+1e-8))
    denominator = (np.abs((target+1e-8)) + np.abs((pred+1e-8))) / 2
    return np.mean((numerator+1e-8) / (denominator+1e-8)) * 100

def mse_metric(pred,target):
    mse_loss=np.mean(((target) - (pred)) ** 2)  # Compute loss
    return mse_loss

def mae_metric(pred, target):
    mae = np.mean(np.abs(target - pred))
    return mae

def r2_score_metric(pred,target):
    target=target+1e-8
    pred=pred+1e-8
    SS_res = np.sum((target - pred) ** 2)
    SS_tot = np.sum((target - np.mean(pred)) ** 2)
    r2_s = 1 - SS_res / SS_tot
    return r2_s




################plotting function-loss,accuracy##############
def plot_loss_epoch(tr_losses, tst_losses, epochs, str_loss_type,plot_filename):
    """
    Plot multiple training and testing loss curves over epochs with improved spacing and readability.

    Parameters:
    tr_losses (list of numpy arrays): List of arrays containing training losses for each epoch.
    tst_losses (list of numpy arrays): List of arrays containing testing losses for each epoch.
    epochs (int): Number of epochs.
    str_loss_type (list of str): List of strings representing the type of loss (e.g., 'MSE', 'Cross-Entropy', etc.).

    Returns:
    None
    """
    x = np.arange(epochs)
    plt.figure(figsize=(12, 12))  # Set the figure size

    for i, (tr_loss, tst_loss) in enumerate(zip(tr_losses, tst_losses)):
        plt.subplot(2, 3, i+1)
        plt.plot(x, tr_loss, label=f'Training Loss {i}', marker='o')  # Plot training loss with circle markers
        plt.plot(x, tst_loss, label=f'Testing Loss {i}', marker='x')  # Plot testing loss with x markers
        # Adding titles and labels
        plt.title(f'Comparison of Training and Testing Losses ({str_loss_type[i]})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        # Add a legend to distinguish the lines
        plt.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=3.0)  # Adds padding between plots and adjusts subplots to fit into figure area.
    #saveplot
    plt.savefig(plot_filename, format='png', dpi=300)
    # Show the plot
    plt.show()


def plot_results(prediction_total,total_target_time,total_target_data,plot_filename,df_columns,features):
    
    x = np.arange(len(total_target_time))
    print(total_target_data[:,26])
    plt.figure(figsize=(12, 12))  # Set the figure size
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        pred_data=prediction_total[:,features[i]]
        print(df_columns[features[i]])
        if df_columns[features[i]]=="rain (mm)":
            pred_data=np.maximum(pred_data, 0)
        plt.plot(x, total_target_data[:,features[i]], label=f'Ground Truth {df_columns[features[i]]}', marker='o')  # Plot training loss with circle markers
        plt.plot(x, pred_data, label=f'Forecast Prediction {df_columns[features[i]]}', marker='x')  # Plot testing loss with x markers
        # Adding titles and labels
        plt.title(f' Feature ({df_columns[features[i]]})')
        plt.xlabel(f'({[len(total_target_time)*10]}) minutes')
        plt.ylabel('Feature Value')
        # Add a legend to distinguish the lines
        plt.legend()
    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=3.0)  # Adds padding between plots and adjusts subplots to fit into figure area.
    #saveplot
    plt.savefig(plot_filename, format='png', dpi=300)
    # Show the plot
    plt.show()


