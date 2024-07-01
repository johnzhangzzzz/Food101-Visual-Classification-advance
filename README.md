食品视觉分类器 🍔👁

此视觉分类模型部署在了huggingface space上可以点击[此处](https://john000z-foodvision-assum.hf.space "https://john000z-foodvision-assum.hf.space")进行使用，初次使用可能需要等待3到5分钟来等待系统唤醒。

我所使用的食品训练数据的是food101数据集，其中包含**101种**常见类型食品（需要注意，由于训练数据有限，此模型并非可以识别任意食品--）如披萨，饺子，炸薯条，炒饭，巧克力慕斯等，每种类别各100张训练图片[详细品类列表详见此处](https://huggingface.co/spaces/john000z/foodvision_assum/blob/main/class_names_chinese.txt). 大家可以根据**品类列表**从网上寻找对应的图片进行上传。

[原项目](https://www.learnpytorch.io/09_pytorch_model_deployment/)在测试集上的精度大约为60%，我这边替换了其中的模型并重新训练了数据，使得精度提到80%以上，同时也对界面进了中文化处理。  




