import pandas as pd
import argparse

#returns percentage of each group in undetected
def ratio_in_undetected(data, col):
    col_classes = data.groupby(col)[col].count()
    return round(col_classes/data[col].count()*100, 2)

#returns percentage of each group in total
def ratio_in_total(data, total_data, col):
     col_classes = data.groupby(col)[col].count()
     print(total_data.groupby(col)[col].count())
     return round(col_classes/total_data.groupby(col)[col].count()*100, 2)

#returns ratio of undetected/total
def ratio(data, total_data):
     return round(data.shape[0]/total_data.shape[0]*100, 2)

def read_files(model):
     train = pd.read_csv("fairface/fairface_label_train.csv")
     val = pd.read_csv("fairface/fairface_label_val.csv")

     #choose rows with service_test = True to get balanced dataset
     train = train[(train['service_test'] == True)] 
     val = val[(val['service_test'] == True)] 

     #read undetected faces 
     undetected_train = pd.read_csv("fairface/"+ model + "/train_balanced.csv")
     undetected_val = pd.read_csv("fairface/"+ model  + "/val_balanced.csv") 

     return undetected_train, train, undetected_val, val

#generate csv file
def gen_csv(param, models):
     train_df, val_df, undetected_percentage_train, undetected_percentage_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
     undetected_number_train, undetected_number_val = pd.DataFrame(),pd.DataFrame()
     param_data_status = False
     i = 1

     
     #for each model store result 
     for model in models:

          #read files with undetected faces
          undetected_train, train, undetected_val, val = read_files(model)

          ratio_in_total(undetected_train, train, param)
          ratio_in_total(undetected_val, val, param )

          res_train = ratio_in_undetected(undetected_train, param).rename().to_frame().reset_index()
          res_val = ratio_in_undetected(undetected_val, param).rename().to_frame().reset_index()
          
          number_train = undetected_train[param].value_counts(sort = False).rename().to_frame().reset_index()
          number_val = undetected_val[param].value_counts(sort = False).rename().to_frame().reset_index()
          if param_data_status == False:
               train_df.insert(0, param, res_train[param])
               val_df.insert(0, param, res_val[param])
               undetected_number_train.insert(0, param, number_train['index'])
               undetected_number_val.insert(0, param, number_val['index'])
               param_data_status = True
          

          res_train.columns = [param, '%']
          res_val.columns = [param, '%']

          train_df.insert(i, model, res_train["%"])
          val_df.insert(i, model, res_val["%"])
          
         
          undetected_number_train.insert(i, model, number_train[0])
          undetected_number_val.insert(i, model, number_val[0])
          # ratio_train = ratio(undetected_train, train)
          # ratio_val = ratio(undetected_val, val)
          # undetected_percentage_train = undetected_percentage_train.append({model : ratio_train}, ignore_index =True)
          # undetected_percentage_val = undetected_percentage_val.append({model : ratio_val}, ignore_index = True)

          
          i+=1

     train_df.to_csv("results_csv/" + param + "_train.csv")
     val_df.to_csv("results_csv/" + param + "_val.csv")
     undetected_number_train.to_csv("results_csv/" + param + "_number_train.csv")
     undetected_number_val.to_csv("results_csv/" + param + "_number_val.csv")
     # undetected_percentage_train.to_csv("results_csv/" + "undetected_percentage_train.csv")
     # undetected_percentage_val.to_csv("results_csv/" + "undetected_percentage_val.csv")
if __name__ == '__main__':
     columns = ["age", "gender", "race"]
     models = ["AnyFace", "Yolov5l", "retinaface", "opencv", "ssd", "mtcnn", "dlib" ]
     #, "Yolov5l", "retinaface", "opencv", "ssd", "mtcnn", "dlib"
     parser = argparse.ArgumentParser()
     parser.add_argument('--model_name', type=str, default = 'retinaface', help = 'model name')
     parser.add_argument('--stats_format', type=str, default = 'csv', help = 'model name')
     parser.add_argument('--param', type = str, default = 'age', help = 'age, gender or race')
     opt = parser.parse_args()

     gen_csv(opt.param, models)
