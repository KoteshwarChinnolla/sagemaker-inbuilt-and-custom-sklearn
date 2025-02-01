# sagemaker-inbuilt-and-custom-sklearn
A sagemaker deployment of machine learning model using inbuilt and custom sklearn machine learning models

Inbuilt XG-boost algorithm which Sagemaker already has the docker image and all we do is to pull it and perform our task providing test and train data

## Inbuilt code 
We need to specify our ML code as Python. The Sagemaker works when we write the code in the sage training script formation.
>[!NOTE]
> Format includes writing
> 1. Input function
> 2. Output function
> 3. predict function

 [Code is in train.py](https://github.com/KoteshwarChinnolla/sagemaker-inbuilt-and-custom-sklearn/blob/main/train.py)
