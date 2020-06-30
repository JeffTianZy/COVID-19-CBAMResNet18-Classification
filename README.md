# COVID-19-CBAMResNet18-Classification
A simple network module for pneumonia x-ray classification
+ Overall Accuracy: 96.8%
+ COVID-19 Recall/Precistion: 100%
## Abstract
A three-category classifier for pneumonia x-ray that distiguish non-pneumonia, normal pneumonia and COVID-19. We added CBAM (Convolutional Block Attention Module) before the first layer and after the last conbolutional layer of ResNet18 that significantly enhanced its ability.
## Datasets
**Dataset 1**: covid-chestxray-dataset(only COVID-19 data)

https://github.com/ieee8023/covid-chestxray-dataset

**Dataset 2**: CoronaHack-Chest X-Ray-Dataset(only normal pneumonia and non-pneumonia data)

https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

**Integration pack**: (contains all data above)

BaiduNetdisk: https://pan.baidu.com/s/11BwGCB1n2rQvHGc137_oLg 

Password: dw7a
## Experiments
**ResNet18 Classfication**

We trained a ResNet18 model on the previous dataset by finetuning weights from Imagenet **(Batch_size=24, Epoch=50, optim=Adam, learning_rate=0.001, criterion=CrossEntropy)**.

Results are as follows:

ResNet18|Precision|Recall|F1-score|Num
:----:|:----:|:----:|:----:|:----:
Normal|0.94|0.88|0.91|151
Pneumonia|0.95|0.99|0.97|411
COVID-19|1.00|0.70|0.82|23

**CBAMResNet18 Classfication**

We then added CBAM to the network with the same hyperparameter.

Results are as follows:

CBAMResNet18|Precision|Recall|F1-score|Num
:----:|:----:|:----:|:----:|:----:
Normal|0.93|0.95|0.94|151
Pneumonia|0.98|0.97|0.98|411
COVID-19|1.00|1.00|1.00|23

## Visualization
By using Grad-CAM, we can visualize the contribution of CBAM (in folder)
