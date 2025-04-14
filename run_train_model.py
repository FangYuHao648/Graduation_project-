import os
import yaml
from QtFusion.path import abs_path
from tools.train import main
import warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

if __name__ == '__main__':
    workers = 1
    batch = 2

    data_name = "Qrcode"
    data_default = f'datasets/{data_name}/{data_name}.yaml'
    name_default = f'train_v6_{data_name}'

    data_path = abs_path(data_default, path_type='current')  # 数据集的yaml的绝对路径
    unix_style_path = data_path.replace(os.sep, '/')
    # 获取目录路径
    directory_path = os.path.dirname(unix_style_path)
    # 读取YAML文件，保持原有顺序
    with open(data_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # 修改path项
    if 'path' in data:
        data['path'] = directory_path
        data['train'] = f'datasets/{data_name}/images/train'
        data['val'] = f'datasets/{data_name}/images/valid'
        data['test'] = f'datasets/{data_name}/images/test/'

        # 将修改后的数据写回YAML文件
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)

    main(data_path=data_default, name=name_default, workers=workers,
         batch_size=batch, output_dir="runs/detect", epochs=100)
