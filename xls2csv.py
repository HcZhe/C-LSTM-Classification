import pandas as pd
import os

def convert_xls_to_csv(folder_path):
    # 获取目录下所有.xls文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.xls')]

    # 逐个文件进行处理
    for file_name in files:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)

        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 构建CSV文件名（替换扩展名）
        csv_file_path = os.path.join(folder_path, file_name.replace('.xls', '.csv'))

        # 保存为CSV文件，编码格式为UTF-8
        df.to_csv(csv_file_path, index=False, encoding='utf-8')


# 指定要转换的文件夹路径
folder_path = r'C:\Users\12307\Desktop\Test\class_1'
convert_xls_to_csv(folder_path)
