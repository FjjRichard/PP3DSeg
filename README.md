# PP3DSeg
PP3DSeg这个工具是基于PaddlePaddle和PaddleSeg构建的，其中3DUnet网络和一些transform方法是参考https://github.com/wolny/pytorch-3dunet 。

整个项目基本和PaddleSeg很像，只是针对3D医疗数据进行修改。专门用来处理3D的医疗数据，并对数据进行3D分割。

目前项目中只支持3DUnet网络和交叉熵损失函数。

数据增强暂时支持随机水平翻转、随机垂直翻转，重采样，归一化，随机对比度改变，随机角度旋转等方法。

可见PP3DSeg这个项目是多么不成熟的。里面存在各种各样的未知Bug，不过后面会慢慢更新，加入更多处理医疗数据的方法和3D分割网络等等。希望更多大佬可以给予珍贵的意见，在这表示非常感激！