# Image-Classifier-Application

## Project Description:
An image classifier application trained on images of flowers and runs from the command line. Over 102 types of flowers are found within the flowers dataset. A neural network will be trained using a pretrained model and a custom classifier to name each flower. The application allows for at least two different possible pretrained models to be used. 

<p align="center">
  <img src="Image-Classifier-Application\assets\Flowers.png">
</p>

## Installations: 
Ensure the following packages are installed in order to run the project:
- Matplotlib
- Numpy
- PyTorch
- Torchvision
- Json
- Pil

## How to Interact with the Project: 
In order to train the model, run the following command from the command line in order to train the model: 
```
python train.py
```
In order to make a prediction, add an image to the image_test folder and run the following command: 
```
python predict.py
```
 
After every command, questions will be asked and require user input. Answer each question in order to build a custom model. 