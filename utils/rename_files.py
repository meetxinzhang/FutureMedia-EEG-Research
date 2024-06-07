import os


def rename_folders(directory):
    # 获取目录下的所有文件夹
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # 遍历文件夹并重命名
    for folder in folders:
        if folder.startswith('2024-'):
            old_name = os.path.join(directory, folder)
            new_name = os.path.join(directory, folder.replace(':', '_'))  # 在文件夹名称前添加"new_"
            os.rename(old_name, new_name)
            print(f"Renamed folder: {old_name} to {new_name}")


# 指定目录路径
directory_path = "/data1/zhangxin/data1/zhangxin/GitHub/FutureMedia-EEG-Research/log/2024-PD-table--syncnet-t_dff"

# 调用函数进行重命名
rename_folders(directory_path)