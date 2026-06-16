
import configparser

if __name__ == '__main__':
    config = configparser.RawConfigParser()
   
    # Section represents the file you want to set for the 'parameters' 
    # 0 represents for the file: Joint_preprocessing.py
    # 1 represents for the file: data.py
    # 2 represents for the file: Unet.py
    # 3 represents for the file: run_model.py
    section = 3

    experiment_name = 'DIP'

    if section == 0:
        file_name = experiment_name+'_Joint_Preprocessing.cfg'
        config.add_section('Joint Preprocessing')
        config.set('Joint Preprocessing', 'image_dimension', '448')
        config.set('Joint Preprocessing', 'input_directory','/Users/jiangxiaohan/Desktop/Code/output')
        config.set('Joint Preprocessing', 'output_directory', '/Users/jiangxiaohan/Desktop/Code/predict')

    if section == 1:
        file_name = experiment_name+'Code/_Data_Conversion.cfg'
        config.add_section('Data Conversion')
        config.set('Data Conversion', 'image_dimension', '448')
        config.set('Data Conversion', 'train_data_path','/Users/jiangxiaohan/Desktop/Data/train')
        config.set('Data Conversion', 'test_data_path','/Users/jiangxiaohan/Desktop/Data/test')
        config.set('Data Conversion', 'validation_data_path','/Users/jiangxiaohan/Desktop/Data/val')
        config.set('Data Conversion', 'data_augmentation','false') 
        config.set('Data Conversion', 'image_dimension','448')
        config.set('Data Conversion', 'train_image_npy',experiment_name+'_train_image.npy') 
        config.set('Data Conversion', 'train_mask_npy',experiment_name+'_train_mask.npy') 
        config.set('Data Conversion', 'train_id_npy',experiment_name+'_train_id.npy') 
        config.set('Data Conversion', 'test_image_npy',experiment_name+'_test_image.npy') 
        config.set('Data Conversion', 'test_id_npy',experiment_name+'_test_id.npy') 
        config.set('Data Conversion', 'validation_data_image_npy',experiment_name+'') 
        config.set('Data Conversion', 'validation_data_mask_npy','') 
        config.set('Data Conversion', 'validation_data_id_npy','') 

    if section == 2:
        file_name = experiment_name+'_Train.cfg'
        config.add_section('Training')
        config.set('Training','train_image_npy',experiment_name+'_train_image.npy')
        config.set('Training','train_mask_npy',experiment_name+'_train_mask.npy')
        config.set('Training','test_image_npy',experiment_name+'_test_image.npy')
        config.set('Training','test_mask_npy',experiment_name+'_test_mask.npy')
        config.set('Training','test_id_npy',experiment_name+'_test_id_npy')
        config.set('Training','val_image_npy',experiment_name+'_val_image.npy')
        config.set('Training','val_mask_npy',experiment_name+'_val_mask.npy')
        config.set('Training','weight_file_name',experiment_name+'_weights.h5')
        config.set('Training','train_only','true')
        config.set('Training','train_and_predict','false')
        config.set('Training','train_and_predict_with_val','false')
        config.set('Training','prediction_directory', 'pred_dir')

    if section == 3:
        file_name = experiment_name+'_Predict.cfg'
        config.add_section('Prediction')
        config.set('Prediction','image_dimension',448)
        config.set('Prediction', 'weight_file_name', '/Users/jiangxiaohan/Desktop/Code/weights.h5')
        config.set('Prediction', 'test_image_npy', 'imgs_test.npy')
        config.set('Prediction', 'test_id_npy','imgs_id_test.npy') 
        config.set('Prediction','prediction_directory', '/Users/jiangxiaohan/Desktop/predict')

    with open(file_name,'w') as configfile:
        config.write(configfile)
