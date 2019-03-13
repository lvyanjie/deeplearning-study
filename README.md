## 基于keras深度学习框架实现（后续持续更新）
  
### classification
- Densenet
- Inception-v1
- Inception-v2
- Inception-v2
- Inception-v3
- Inception-v4
- Inception-resnet-v2
- ResNet50
- ResNext50
- VGG16
- Xception

### 模型验证(classificaiton) 
模型验证.ipynb  
需要将模型代码整合到一个py文件中，将模型代码文件导入到脚本中，如：  
```
import sys
sys.path.append('./models/resnext50.py')
```
导入后，直接调用模型构建函数，构建相应的模型，如：
```
from models.resnext50 import ResNext50
model = ResNext50(input_shape=(224,224,3), classes = len(data_label))
```
最后，直接run all cells进行模型训练

### 验证数据
pascal dataset