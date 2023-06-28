import pandas as pd
import argparse

def read_file(file_name):
    return pd.read_csv(file_name)

def count_groups(df, group):
    return df[group].value_counts(sort=False).rename().to_frame().reset_index().sort_values(by='index')

def gen_csv(folders, file_name, groups):
    for group in groups:
        result = pd.DataFrame()

        for folder in folders:
            file_path = folder + "/" + file_name
            df = read_file(file_path)

            count = count_groups(df, group)

            if group not in result.columns:
                result[group] = count['index']
            
            result[folder] = count[0].values  # Use .values to assign the counts based on position instead of index alignment
            
        result.to_csv("results/" + f"{group}_result.csv", index=False)

if __name__ == '__main__':
    groups = ["age", "gender", "race"]
    folders = [ "fairface/retinaface", "fairface/opencv", "fairface/dlib", "fairface/ssd", "fairface/mtcnn"]
    file_name = "train_balanced.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_names', type=str, nargs='+', default = folders, help = 'folder names')
    parser.add_argument('--file_name', type=str, default = file_name, help = 'file name')
    parser.add_argument('--groups', type=str, nargs='+', default = groups, help = 'age, gender or race')
    opt = parser.parse_args()

    gen_csv(opt.folder_names, opt.file_name, opt.groups)
