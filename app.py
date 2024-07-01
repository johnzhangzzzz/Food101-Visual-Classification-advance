### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model,create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names

with open("class_names.txt", "r") as f: # reading them in from class_names.txt
    class_names_0= [food_name.strip() for food_name in  f.readlines()]

with open("class_names_chinese.txt", "r") as f: # reading them in from class_names.txt
    class_names_1 = [food_name.strip() for food_name in  f.readlines()]

class_names=[]
for _ in range(len(class_names_0)):
    class_names.append(class_names_0[_]+', '+class_names_1[_])    
### 2. Model and transforms preparation ###    

# Create model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=101, # could also use len(class_names)
)
vit, vit_transforms = create_vit_model(
    num_classes=101, # could also use len(class_names)
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="pretrained_effnetb2_feature_extractor_food101_100_percent.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)
vit.load_state_dict(
    torch.load(
        f="pretrained_vit_feature_extractor_food101_100_percent.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img_effnetb2 = effnetb2_transforms(img).unsqueeze(0)
    img_vit = vit_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax((effnetb2(img_effnetb2)+vit(img_vit))/2, dim=1)
        
    #with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        #pred_probs_vit = torch.softmax(vit(img), dim=1)
    
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
#title = "FoodVision Big 🍔👁"
description = "分类器使用的训练数据的是food101数据集，其中包含101种不同类型食品，\
    如披萨，饺子，炸薯条，炒饭，巧克力慕斯等，\
    每种类别各100张训练图片\
    [详细品类列表详见此处](https://huggingface.co/spaces/john000z/foodvision_assum/blob/main/class_names_chinese.txt). \
    大家可以根据**品类列表**从网上寻找对应的图片进行识别"
    
    
article = "[原项目](https://www.learnpytorch.io/09_pytorch_model_deployment/)在测试集上的精度大约为60%，\
    我这边替换了其中的模型并重新训练了数据，使得精度提到80%以上，同时也对界面进了中文化处理"

title = "食品分类器 🍔👁"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="预测结果"), #Predictions
        gr.Number(label="预测消耗时间（s）"), #Prediction time (s)
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Launch the app!
demo.launch()