"""

    训练 LSTM 模型
    使用 提供的训练集

"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt




from lstm_model import lstmNet


def deal_train_dataset(df):

    # 按照主机名进行分组
    df_group = df.groupby("vmid")
    
    x = 0
    # 处理每个主机
    for vmid, group in df_group:
    
        # 对时间戳进行排序, 升序: ascending = True
        group_sort = group.sort_values("timestamp", ascending = True)
        
        num_cnt = len(group_sort)
        print("host_name: {}, size: {}".format(vmid, num_cnt))
        
        if num_cnt < 100:
        
            train_data = np.array(group_sort) #先将数据框转换为数组
            train_data_list = train_data.tolist()  #其次转换为列表
        
            print(train_data_list)
            group_sort.plot("timestamp", "mean_cpu", kind = "line")
            plt.show()
            break
        
    
    
    
    return 0


# 训练模型入口
if __name__ == '__main__':

    # 数据集路径
    dataset_file = "cpu_usage_trends_predictions/dataset/cpu_train.csv"

    # 加载模型
    lstm = lstmNet()
    print("")
    print(lstm)

    # 读取数据集
    if not os.path.exists(dataset_file):
        print("dataset not exists:", dataset_file)
        exit(1)

    """
        读取数据集
        header:     不读取列名
        names:      定义列名
        index_col:  作为索引的列
        usecols:    使用的列, 其他列舍去
    """
    df = pd.read_csv(dataset_file,
                     header = None,
                     names = ["timestamp", "vmid", "min_cpu", "max_cpu", "mean_cpu"],
                     index_col = ["vmid"],
                     usecols = ["timestamp", "vmid", "mean_cpu"])
    
    print("")
    print(df)
    
    
    # 处理数据集
    ret = deal_train_dataset(df)

    # 训练


    # 导出训练结果
