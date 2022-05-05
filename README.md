# MS-EMD: Few-shot font style transfer with multiple style encoders
### 环境配置
TensorFlow=2.3
Python=3.8
pygame
opencv-python
tqdm

### 1. 生成字体图片
将字体文件放入'./fonts/fonts_243'路径下， 运行font2img.py，生成80×80的图片：
```
python font2img.py
```
将font2img.py中96行和第99行分别改为：
```
img_path = 'images_resol_256'
resol = 256
```
再次运行font2img.py，生成256×256的图片。
### 2. 生成数据集
运行get_data_list.py，生成80×80的训练数据：
```
python get_data_list.py
```
将get_data_list.py中第18行、第20行、第21行分别改为：
```
image_path = base_path + '/images_resol_256'
train_save_path = base_path + '/train_256'
test_save_path = base_path + '/test_256'
```
再次运行get_data_list.py，生成256×256的训练数据。
### 3. 训练模型
**阶段1 **指定GPU，运行train.py，训练EMD模型：

```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --input_dir datasets/243/train --output_dir results/243/train --save_summary 1 --style_num 3 --with_dis 1
```
**阶段2** 再次运行train.py，训练local enhancer：

```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --input_dir datasets/243/train_256 --output_dir results/243/train_256 --save_summary 1 --epochs 50 --input_size 256 --batch_size 16 --save_size 2 8 --style_num 3 --with_local_enhancer 1 --global_checkpoint results/243/train
```
### 4. 测试模型
指定GPU，运行test.py：
```
CUDA_VISIBLE_DEVICES=0 python src/test.py --input_dir datasets/243/test_256 --output_dir results/243/test_256 --checkpoint results/243/train_256 --input_size 256 --style_num 3 --with_local_enhancer 1
```
运行test.py，生成融合结果：
```
CUDA_VISIBLE_DEVICES=0 python src/test.py --input_dir datasets/243/test_256 --output_dir results/243/interp_256 --checkpoint results/243/train_256 --input_size 256 --batch_size 5 --save_size 5 1 --style_num 3 --with_local_enhancer 1 --interp 1
