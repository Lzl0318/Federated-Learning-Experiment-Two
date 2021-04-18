import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for c in [r'\2_1_', r'\3_1_', r'\4_1_', r'\2_1_new_', r'\3_1_new_', r'\4_1_new_']:
    # df = pd.DataFrame(columns=['acc', 'loss', 'epoch'], index=['bob', 'alice', 'local', 'fed'])
    with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\citeseer-results\testing'+c+'1'+'.pkl', 'rb') as f1:
        df1 = pkl.load(f1)
    f1.close()
    with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\citeseer-results\testing'+c+'2'+'.pkl', 'rb') as f2:
        df2 = pkl.load(f2)
    f2.close()
    with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\citeseer-results\testing'+c+'3'+'.pkl', 'rb') as f3:
        df3 = pkl.load(f3)
    f3.close()
    with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\citeseer-results\testing'+c+'4'+'.pkl', 'rb') as f4:
        df4 = pkl.load(f4)
    f4.close()
    with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\citeseer-results\testing'+c+'5'+'.pkl', 'rb') as f5:
        df5 = pkl.load(f5)
    f5.close()
    df = (df1+df2+df3+df4+df5)/5
    print(df)
    




