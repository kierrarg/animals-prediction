# Animals Prediction

This is my take on the cats vs dogs prediction problem wherein I include ferrets.

## Running this program

You must first run:
```bash
python -m pip install -r requirements.txt
```
to install the requirmenets.

## How it works

This project develops a baseline cnn model and uses a multi vgg block model, each layer uses the ReLU activation function.

I have also implemented dropout regularization, the performance of this project could be increased by including image data augmentation or modifying the image sizes so that they are all the same size.

## Results
The number shown at the bottom of the output indicates the accuracy in classification.

### Cats vs. Dogs
![cat + dog](https://github.com/kierrarg/animals-prediction/assets/78625238/02b68cec-9d9e-44ac-b6da-4a84c93f3075)
![cat + dog 1](https://github.com/kierrarg/animals-prediction/assets/78625238/1444bfbe-fccf-474e-be08-d6caa6073375)

### Dogs vs. Ferrets
![dog + ferret](https://github.com/kierrarg/animals-prediction/assets/78625238/81bfcf89-3d95-429b-893d-1b839e881bc8)
![dog + ferret 1](https://github.com/kierrarg/animals-prediction/assets/78625238/de3bde38-3c53-402a-b8cd-279ef65a2dae)

### Cats vs. Ferrets
![cat + ferret](https://github.com/kierrarg/animals-prediction/assets/78625238/82cca505-c6ee-409c-84c6-97679a2bb6e2)
![cat + ferret 1](https://github.com/kierrarg/animals-prediction/assets/78625238/99402090-20f8-4e1f-83dd-020494dc9c28)

### Cats vs. Dogs vs. Ferrets
![dog + cat + ferret](https://github.com/kierrarg/animals-prediction/assets/78625238/0634b53d-2535-4889-9bb5-66636264ae59)
![dog + cat + ferret 1](https://github.com/kierrarg/animals-prediction/assets/78625238/f8e4f076-6a3b-4f26-9d33-951ea65e5a6a)
