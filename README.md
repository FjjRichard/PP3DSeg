# PP3DSeg

PP3DSeg这个工具是基于PaddlePaddle和PaddleSeg构建的，其中3DUnet网络和一些transform方法是参考https://github.com/wolny/pytorch-3dunet 。

整个项目基本和PaddleSeg很像，只是针对3D医疗数据进行修改。专门用来处理3D的医疗数据，并对数据进行3D分割。

目前项目中只支持3DUnet网络和交叉熵损失函数。

数据增强暂时支持随机水平翻转、随机垂直翻转，重采样，归一化，随机对比度改变，随机角度旋转等方法。



## 运行环境

1.建议在飞桨AIStudio平台在线运行

[【PP3DSeg】基于PaddleSeg实现3DUnet分割肝脏](https://aistudio.baidu.com/aistudio/projectdetail/2549429)

2.本地环境

- Python 3.7 +

- PaddlePaddle >=2.2.1

- Nvidia 显卡 

## 快速使用

### 1.安装依赖

```
git clone https://github.com/richarddddd198/PP3DSeg.git
cd PP3DSeg
pip install -r requirements.txt
```

### 2. 准备数据

数据文件结构如下，文件夹名和文件名可以自行定义。

```
./dataset/  # 数据集根目录
|--images  # 原图目录
|  |--xxx1.nii (xx1.nii.gz)
|  |--...
|  └--...
|
|--mask  # 标注图目录
|  |--xxx1.nii (xxx1.nii.gz)
|  |--...
|  └--...
```

### 3.划分数据

按一定比例划分数据，并生成文件列表train.txt/val.txt

`train.txt`，`val.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为标注图像文件相对于dataset的相对路径。如下所示：

```
images/xxx1.nii (xx1.nii.gz) mask/xxx1.nii(xxx1.nii.gz)
images/xxx2.nii (xx2.nii.gz) mask/xxx2.nii(xxx1.nii.gz)
...
```

### 4.创建DataSet

```
from PP3DSeg.transforms import transforms as T
from PP3DSeg.datasets import Dataset
WW = 350 #设置窗宽窗位
WC = 60
SIZE = (48,256,256) # 设置重采样参数

#使用数据增广
train_transforms =T.Compose( [
    T.Normalize(ww=WW,wc=WC),
    T.RandomHorizontalFlip(),#水平翻转
    T.RandomContrast( alpha=(0.2, 1.6)),#随机改变对比度
    T.RandomRotate(max_rotation=25),#随机旋转一定角度
    T.Resize3D(target_size=SIZE),#重采样
    T.ToTensor()
])

val_transforms = T.Compose([
    T.Normalize(ww=WW,wc=WC),
    T.Resize3D(target_size=SIZE),
    T.ToTensor()
])

#文件列表文件路径
train_path = '/home/aistudio/work/liver/train_list.txt'
val_path = '/home/aistudio/work/liver/val_list.txt'
# 创建DataSet
train_dataset = Dataset(
                transforms=train_transforms,
                dataset_root='/home/',
                num_classes=2,  #分类类别
                mode='train',  #用于训练的DataSet
                train_path=train_path,
                flipud=False) #是否上下翻转
val_dataset = Dataset(
                transforms=val_transforms,
                dataset_root='/home/',
                num_classes=2,
                mode='val',
                val_path=val_path,
                flipud=False)
```

### 5.导入3D分割网络

```
from PP3DSeg.models.unet3d import Unet3d
model = Unet3d(class_num=2)
```

### 6.设置优化器和损失函数

```
from PP3DSeg.models.losses.cross_entropy_loss import CrossEntropyLoss
lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.02, verbose=False)
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters())

losses = {}
losses['types'] = [CrossEntropyLoss()] 
losses['coef'] = [1]
```

### 7.开始训练

```
from PP3DSeg.core import train
train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='./Output', #模型保存路径
    iters=1000,
    batch_size=2,
    save_interval=10, #模型保存间隔
    log_iters=2,  #日志打印间隔
    losses=losses,
    use_vdl=True)
```

### 8.验证

```
from PP3DSeg.core import evaluate
evaluate(model,val_dataset)
```

## 参考资料

- [https://github.com/PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

- [https://github.com/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)

可见PP3DSeg这个项目是多么不成熟的。里面存在各种各样的未知Bug，不过后面会慢慢更新，加入更多处理医疗数据的方法和3D分割网络等等。希望更多大佬可以给予珍贵的意见，在这表示非常感激！