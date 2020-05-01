1. Project overview and objectives 
The main purpose of this project was to build a CNN model that would classify if subject has a tumor or not base on MRI scan. I used the VGG-16, Inception v3 , xception model architecture and weights to train the model for this binary problem. I used accuracy as a metric to justify the model performance which can be defined as:

Accuracy=Number of correclty predicted images/Total number of tested images×100%

Final results look as follows:

| __Set__ | __Accuracy__ |
|-------------|------------
| Validation Set         | ~92%     |
| Test Set         | ~92% |

1.1. Data Set Description

The image data that was used for this problem is Brain MRI Images for Brain Tumor Detection. It conists of MRI scans of two classes:

    NO - no tumor, encoded as 0
    YES - tumor, encoded as 1

Unfortunately, the data set description doesn't hold any information where this MRI scans come from and so on.

1.2. What is Brain Tumor?

A brain tumor occurs when abnormal cells form within the brain. 
There are two main types of tumors: cancerous (malignant) tumors and benign tumors. Cancerous tumors can be divided into primary tumors, which start within the brain, and secondary tumors, which have spread from elsewhere, known as brain metastasis tumors. 
All types of brain tumors may produce symptoms that vary depending on the part of the brain involved. These symptoms may include headaches, seizures, problems with vision, vomiting and mental changes. The headache is classically worse in the morning and goes away with vomiting. Other symptoms may include difficulty walking, speaking or with sensations.
As the disease progresses, unconsciousness may occur.

![Brain_tumor_description](https://user-images.githubusercontent.com/23243761/80807481-9b5f1100-8bbd-11ea-8f44-0f78978943fd.jpg)


2. Setting up the project
For this project I used tensorflow, keras and I used Vgg19 model from the keras and for visulizing the generated data I used matplotlib and plotly 
The dataset which I used for this project provided by Kaggle and you can download it from https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
once you download the dataset from kaggle you need to devide it into the following structure:
```bash
├── brain_tumor_dataset
│   ├── no
│   └── yes
├── Pretrained_models
├── preview
├── TEST
│   ├── NO
│   └── YES
├── TEST_CROP
│   ├── NO
│   └── YES
├── TRAIN
│   ├── NO
│   └── YES
├── TRAIN_CROP
│   ├── NO
│   └── YES
├── VAL
│   ├── NO
│   └── YES
└── VAL_CROP
    ├── NO
    └── YES
```

once you done the separation of the data in order to better understanding of the dataset you could check the following figures which shows that the MRL pictures is categorized either no tumor or mri picture shows the tumor
![no_cancer](https://user-images.githubusercontent.com/23243761/80808050-f5aca180-8bbe-11ea-8f57-f47fab7283ab.png)
![cancer](https://user-images.githubusercontent.com/23243761/80808047-f47b7480-8bbe-11ea-8dfe-49ca269869f1.png)

As you can see, images have different width and height and diffent size of "black corners". Since the image size for VGG-16 imput layer is (224,224) some wide images may look weird after resizing. Histogram of ratio distributions (ratio = width/height):

![distribution](https://user-images.githubusercontent.com/23243761/80808170-3d332d80-8bbf-11ea-8fa6-303f772f9c42.png)

The first step of "normalization" would be to crop the brain out of the images. Let's look at example what this function will do with MRI scans:

![model](https://user-images.githubusercontent.com/23243761/80808303-8d11f480-8bbf-11ea-9f55-2991a92b632c.png)


4. CNN Model

I was using Transfer Learning with VGG-16 architecture and weights as a base model.
4.1. Data Augmentation

Since I had small data set I used the technique called Data Augmentation which helps to "increase" the size of training set.
4.1.1. Demo

That's the example from one image how does augmentation look like.
![original](https://user-images.githubusercontent.com/23243761/80808385-ba5ea280-8bbf-11ea-8eb4-9b93358550df.png)
![augmented_from_original](https://user-images.githubusercontent.com/23243761/80808390-bfbbed00-8bbf-11ea-81dc-2e8c66709a80.png)

The model structure is: 
```bash
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 7, 7, 512)         14714688  
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 25088)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 25089     
=================================================================
Total params: 14,739,777
Trainable params: 25,089
Non-trainable params: 14,714,688
```

here is the log from the computation that have been done:
```bash
Epoch 1/30
50/50 [==============================] - 487s 10s/step - loss: 4.2269 - accuracy: 0.6298 - val_loss: 1.7208 - val_accuracy: 0.7437
/home/eddie/.local/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning:

Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,val_accuracy,loss,accuracy

Epoch 2/30
50/50 [==============================] - 490s 10s/step - loss: 3.4299 - accuracy: 0.6847 - val_loss: 2.6733 - val_accuracy: 0.7722
Epoch 3/30
50/50 [==============================] - 482s 10s/step - loss: 2.4713 - accuracy: 0.7448 - val_loss: 2.2775 - val_accuracy: 0.7880
Epoch 4/30
50/50 [==============================] - 496s 10s/step - loss: 1.9341 - accuracy: 0.7843 - val_loss: 1.0086e-10 - val_accuracy: 0.7980
Epoch 5/30
50/50 [==============================] - 526s 11s/step - loss: 1.7740 - accuracy: 0.7936 - val_loss: 2.3855 - val_accuracy: 0.7785
Epoch 6/30
50/50 [==============================] - 491s 10s/step - loss: 1.5151 - accuracy: 0.8265 - val_loss: 0.6203 - val_accuracy: 0.8006
Epoch 7/30
50/50 [==============================] - 598s 12s/step - loss: 1.2918 - accuracy: 0.8351 - val_loss: 1.1182 - val_accuracy: 0.8354
Epoch 8/30
50/50 [==============================] - 528s 11s/step - loss: 1.5553 - accuracy: 0.8447 - val_loss: 0.0137 - val_accuracy: 0.8411
Epoch 9/30
50/50 [==============================] - 604s 12s/step - loss: 1.0989 - accuracy: 0.8494 - val_loss: 0.0672 - val_accuracy: 0.8449
Epoch 10/30
50/50 [==============================] - 555s 11s/step - loss: 1.2725 - accuracy: 0.8677 - val_loss: 1.0001 - val_accuracy: 0.8513
Epoch 11/30
50/50 [==============================] - 649s 13s/step - loss: 0.8031 - accuracy: 0.8868 - val_loss: 1.1851 - val_accuracy: 0.8829
Epoch 12/30
50/50 [==============================] - 616s 12s/step - loss: 0.8832 - accuracy: 0.8755 - val_loss: 4.8322e-06 - val_accuracy: 0.8808
Epoch 13/30
50/50 [==============================] - 577s 12s/step - loss: 0.6841 - accuracy: 0.9009 - val_loss: 2.0025 - val_accuracy: 0.8956
Epoch 14/30
50/50 [==============================] - 632s 13s/step - loss: 0.6669 - accuracy: 0.8966 - val_loss: 1.3689 - val_accuracy: 0.9241
Epoch 15/30
50/50 [==============================] - 658s 13s/step - loss: 0.7030 - accuracy: 0.9038 - val_loss: 0.8344 - val_accuracy: 0.9367
Epoch 16/30
50/50 [==============================] - 912s 18s/step - loss: 0.5709 - accuracy: 0.9140 - val_loss: 1.8708e-09 - val_accuracy: 0.9007
Epoch 17/30
50/50 [==============================] - 687s 14s/step - loss: 0.5367 - accuracy: 0.9089 - val_loss: 0.5395 - val_accuracy: 0.9399
Epoch 18/30
50/50 [==============================] - 751s 15s/step - loss: 0.5036 - accuracy: 0.9275 - val_loss: 0.8805 - val_accuracy: 0.9430
Epoch 19/30
50/50 [==============================] - 776s 16s/step - loss: 0.4639 - accuracy: 0.9306 - val_loss: 1.7687 - val_accuracy: 0.9367
Epoch 20/30
50/50 [==============================] - 742s 15s/step - loss: 0.7759 - accuracy: 0.9219 - val_loss: 5.1726e-08 - val_accuracy: 0.9205
Epoch 21/30
50/50 [==============================] - 869s 17s/step - loss: 0.7014 - accuracy: 0.9262 - val_loss: 1.9262 - val_accuracy: 0.9209
Epoch 22/30
50/50 [==============================] - 798s 16s/step - loss: 0.7257 - accuracy: 0.9255 - val_loss: 1.8456 - val_accuracy: 0.9430
Epoch 23/30
50/50 [==============================] - 763s 15s/step - loss: 0.4235 - accuracy: 0.9393 - val_loss: 0.0148 - val_accuracy: 0.9367
Epoch 24/30
50/50 [==============================] - 731s 15s/step - loss: 0.5044 - accuracy: 0.9378 - val_loss: 1.5106e-11 - val_accuracy: 0.9205
Epoch 25/30
50/50 [==============================] - 658s 13s/step - loss: 0.4904 - accuracy: 0.9393 - val_loss: 0.0045 - val_accuracy: 0.9241
Epoch 26/30
50/50 [==============================] - 669s 13s/step - loss: 0.3364 - accuracy: 0.9501 - val_loss: 0.2920 - val_accuracy: 0.9114
Epoch 27/30
50/50 [==============================] - 740s 15s/step - loss: 0.2786 - accuracy: 0.9414 - val_loss: 0.7194 - val_accuracy: 0.9241
Epoch 28/30
50/50 [==============================] - 739s 15s/step - loss: 0.3318 - accuracy: 0.9523 - val_loss: 1.1884e-04 - val_accuracy: 0.9205
Epoch 29/30
50/50 [==============================] - 814s 16s/step - loss: 0.7242 - accuracy: 0.9436 - val_loss: 1.7105 - val_accuracy: 0.8987
Epoch 30/30
50/50 [==============================] - 731s 15s/step - loss: 0.2182 - accuracy: 0.9508 - val_loss: 1.7486 - val_accuracy: 0.9177
```

once the computation is done we need to plot the accuracy vs loss:

![model_accuracy_model_loss](https://user-images.githubusercontent.com/23243761/80808783-92bc0a00-8bc0-11ea-8103-c035d82ba47c.png)


5. Conclusions

This project was a combination of CNN model classification problem (to predict wheter the subject has brain tumor or not) & Computer Vision problem (to automate the process of brain cropping from MRI scans). The final accuracy is much higher than 50% baseline (random guess). However, it could be increased by larger number of train images or through model hyperparameters tuning.
