This project is designed the model for the Skin cancer dataset which is provided by Kaggle and can be downloaded from https://www.kaggle.com/fanconic/skin-cancer-efficientnetb0/data

Once you download the dataset from the mentioned link, the first step will be data visulization and labeling of the data which you could the example of the dataset which we are dealing from the following figure:
![Data_visulization](https://user-images.githubusercontent.com/23243761/80826059-e2f79400-8be1-11ea-9069-e2b2dff398be.png)

After finishing the data preprocessing and data processing we could start to build the model, for this project I used RAdam optimizer and for the weight I used imagenet, the model could see at below:
```bash 
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
efficientnet-b0 (Model)      (None, 7, 7, 1280)        4049564   
_________________________________________________________________
global_average_pooling2d (Gl (None, 1280)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1280)              0         
_________________________________________________________________
batch_normalization (BatchNo (None, 1280)              5120      
_________________________________________________________________
dense (Dense)                (None, 2)                 2562      
=================================================================
Total params: 4,057,246
Trainable params: 4,012,670
Non-trainable params: 44,576
_________________________________________________________________
```

After model definition we could start to doing the training and tesing of the model:
```bash
Epoch 1/30
64/65 [============================>.] - ETA: 1s - loss: 0.6968 - acc: 0.6890
Epoch 00001: val_acc improved from -inf to 0.52539, saving model to weights.best.hdf5
65/65 [==============================] - 88s 1s/step - loss: 0.7019 - acc: 0.6890 - val_loss: 0.7287 - val_acc: 0.5254
Epoch 2/30
64/65 [============================>.] - ETA: 0s - loss: 0.4875 - acc: 0.8049
Epoch 00002: val_acc improved from 0.52539 to 0.65234, saving model to weights.best.hdf5
65/65 [==============================] - 41s 624ms/step - loss: 0.4868 - acc: 0.8045 - val_loss: 0.6507 - val_acc: 0.6523
Epoch 3/30
64/65 [============================>.] - ETA: 0s - loss: 0.3785 - acc: 0.8352
Epoch 00003: val_acc did not improve from 0.65234
65/65 [==============================] - 40s 618ms/step - loss: 0.3755 - acc: 0.8368 - val_loss: 0.7286 - val_acc: 0.6484
Epoch 4/30
64/65 [============================>.] - ETA: 0s - loss: 0.3421 - acc: 0.8513
Epoch 00004: val_acc improved from 0.65234 to 0.65625, saving model to weights.best.hdf5
65/65 [==============================] - 41s 626ms/step - loss: 0.3401 - acc: 0.8517 - val_loss: 0.6715 - val_acc: 0.6562
Epoch 5/30
64/65 [============================>.] - ETA: 0s - loss: 0.2991 - acc: 0.8660
Epoch 00005: val_acc improved from 0.65625 to 0.80469, saving model to weights.best.hdf5
65/65 [==============================] - 42s 642ms/step - loss: 0.2994 - acc: 0.8657 - val_loss: 0.5369 - val_acc: 0.8047
Epoch 6/30
64/65 [============================>.] - ETA: 0s - loss: 0.2831 - acc: 0.8817
Epoch 00006: val_acc improved from 0.80469 to 0.80859, saving model to weights.best.hdf5
65/65 [==============================] - 42s 639ms/step - loss: 0.2815 - acc: 0.8825 - val_loss: 0.4463 - val_acc: 0.8086
Epoch 7/30
64/65 [============================>.] - ETA: 0s - loss: 0.2485 - acc: 0.8880
Epoch 00007: val_acc improved from 0.80859 to 0.83398, saving model to weights.best.hdf5
65/65 [==============================] - 41s 627ms/step - loss: 0.2490 - acc: 0.8873 - val_loss: 0.3867 - val_acc: 0.8340
Epoch 8/30
64/65 [============================>.] - ETA: 0s - loss: 0.2401 - acc: 0.8968
Epoch 00008: val_acc did not improve from 0.83398
65/65 [==============================] - 40s 620ms/step - loss: 0.2400 - acc: 0.8974 - val_loss: 0.3358 - val_acc: 0.8242
Epoch 9/30
64/65 [============================>.] - ETA: 0s - loss: 0.2267 - acc: 0.9027
Epoch 00009: val_acc improved from 0.83398 to 0.88477, saving model to weights.best.hdf5
65/65 [==============================] - 41s 634ms/step - loss: 0.2262 - acc: 0.9023 - val_loss: 0.2479 - val_acc: 0.8848
Epoch 10/30
64/65 [============================>.] - ETA: 0s - loss: 0.2010 - acc: 0.9164
Epoch 00010: val_acc improved from 0.88477 to 0.89453, saving model to weights.best.hdf5
65/65 [==============================] - 41s 634ms/step - loss: 0.2001 - acc: 0.9172 - val_loss: 0.2753 - val_acc: 0.8945
Epoch 11/30
64/65 [============================>.] - ETA: 0s - loss: 0.2013 - acc: 0.9188
Epoch 00011: val_acc did not improve from 0.89453
65/65 [==============================] - 41s 627ms/step - loss: 0.2010 - acc: 0.9191 - val_loss: 0.2753 - val_acc: 0.8828
Epoch 12/30
64/65 [============================>.] - ETA: 0s - loss: 0.1906 - acc: 0.9144
Epoch 00012: val_acc did not improve from 0.89453
65/65 [==============================] - 41s 635ms/step - loss: 0.1897 - acc: 0.9143 - val_loss: 0.2788 - val_acc: 0.8887
Epoch 13/30
64/65 [============================>.] - ETA: 0s - loss: 0.1696 - acc: 0.9287
Epoch 00013: val_acc improved from 0.89453 to 0.91602, saving model to weights.best.hdf5
65/65 [==============================] - 41s 625ms/step - loss: 0.1683 - acc: 0.9293 - val_loss: 0.2446 - val_acc: 0.9160
Epoch 14/30
64/65 [============================>.] - ETA: 0s - loss: 0.1722 - acc: 0.9226
Epoch 00014: val_acc did not improve from 0.91602
65/65 [==============================] - 40s 620ms/step - loss: 0.1715 - acc: 0.9224 - val_loss: 0.2712 - val_acc: 0.8867
Epoch 15/30
64/65 [============================>.] - ETA: 0s - loss: 0.1502 - acc: 0.9384
Epoch 00015: val_acc did not improve from 0.91602
65/65 [==============================] - 40s 609ms/step - loss: 0.1523 - acc: 0.9374 - val_loss: 0.3543 - val_acc: 0.8770
Epoch 16/30
64/65 [============================>.] - ETA: 0s - loss: 0.1730 - acc: 0.9312
Epoch 00016: val_acc did not improve from 0.91602
65/65 [==============================] - 40s 618ms/step - loss: 0.1751 - acc: 0.9298 - val_loss: 0.3620 - val_acc: 0.8555
Epoch 17/30
64/65 [============================>.] - ETA: 0s - loss: 0.2001 - acc: 0.9159
Epoch 00017: val_acc did not improve from 0.91602
65/65 [==============================] - 40s 614ms/step - loss: 0.1979 - acc: 0.9167 - val_loss: 0.3453 - val_acc: 0.8555
Epoch 18/30
64/65 [============================>.] - ETA: 0s - loss: 0.1529 - acc: 0.9305
Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.

Epoch 00018: val_acc did not improve from 0.91602
65/65 [==============================] - 41s 633ms/step - loss: 0.1509 - acc: 0.9315 - val_loss: 0.2320 - val_acc: 0.9121
Epoch 19/30
64/65 [============================>.] - ETA: 0s - loss: 0.1240 - acc: 0.9545
Epoch 00019: val_acc improved from 0.91602 to 0.91992, saving model to weights.best.hdf5
65/65 [==============================] - 40s 613ms/step - loss: 0.1230 - acc: 0.9547 - val_loss: 0.1944 - val_acc: 0.9199
Epoch 20/30
64/65 [============================>.] - ETA: 0s - loss: 0.1060 - acc: 0.9600
Epoch 00020: val_acc did not improve from 0.91992
65/65 [==============================] - 40s 619ms/step - loss: 0.1060 - acc: 0.9601 - val_loss: 0.2319 - val_acc: 0.9199
Epoch 21/30
64/65 [============================>.] - ETA: 0s - loss: 0.0747 - acc: 0.9716
Epoch 00021: val_acc did not improve from 0.91992
65/65 [==============================] - 39s 593ms/step - loss: 0.0746 - acc: 0.9716 - val_loss: 0.1928 - val_acc: 0.9199
Epoch 22/30
64/65 [============================>.] - ETA: 0s - loss: 0.0798 - acc: 0.9702
Epoch 00022: val_acc did not improve from 0.91992
65/65 [==============================] - 39s 597ms/step - loss: 0.0814 - acc: 0.9687 - val_loss: 0.2387 - val_acc: 0.9160
Epoch 23/30
64/65 [============================>.] - ETA: 0s - loss: 0.0670 - acc: 0.9751
Epoch 00023: val_acc did not improve from 0.91992
65/65 [==============================] - 39s 599ms/step - loss: 0.0668 - acc: 0.9754 - val_loss: 0.2369 - val_acc: 0.9199
Epoch 24/30
64/65 [============================>.] - ETA: 0s - loss: 0.0656 - acc: 0.9736
Epoch 00024: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.

Epoch 00024: val_acc did not improve from 0.91992
65/65 [==============================] - 39s 606ms/step - loss: 0.0649 - acc: 0.9740 - val_loss: 0.2091 - val_acc: 0.9160
Epoch 25/30
64/65 [============================>.] - ETA: 0s - loss: 0.0610 - acc: 0.9800
Epoch 00025: val_acc did not improve from 0.91992
65/65 [==============================] - 39s 605ms/step - loss: 0.0608 - acc: 0.9803 - val_loss: 0.2334 - val_acc: 0.9062
Epoch 26/30
64/65 [============================>.] - ETA: 0s - loss: 0.0591 - acc: 0.9770
Epoch 00026: val_acc did not improve from 0.91992
65/65 [==============================] - 39s 605ms/step - loss: 0.0588 - acc: 0.9769 - val_loss: 0.2163 - val_acc: 0.9199
Epoch 27/30
64/65 [============================>.] - ETA: 0s - loss: 0.0608 - acc: 0.9785
Epoch 00027: val_acc did not improve from 0.91992
65/65 [==============================] - 39s 603ms/step - loss: 0.0619 - acc: 0.9779 - val_loss: 0.2306 - val_acc: 0.9102
Epoch 28/30
64/65 [============================>.] - ETA: 0s - loss: 0.0626 - acc: 0.9756
Epoch 00028: val_acc improved from 0.91992 to 0.93359, saving model to weights.best.hdf5
65/65 [==============================] - 40s 619ms/step - loss: 0.0618 - acc: 0.9759 - val_loss: 0.2353 - val_acc: 0.9336
Epoch 29/30
64/65 [============================>.] - ETA: 0s - loss: 0.0529 - acc: 0.9819
Epoch 00029: val_acc did not improve from 0.93359
65/65 [==============================] - 39s 602ms/step - loss: 0.0522 - acc: 0.9822 - val_loss: 0.2451 - val_acc: 0.9121
Epoch 30/30
64/65 [============================>.] - ETA: 0s - loss: 0.0484 - acc: 0.9833
Epoch 00030: val_acc did not improve from 0.93359
65/65 [==============================] - 39s 602ms/step - loss: 0.0481 - acc: 0.9831 - val_loss: 0.2279 - val_acc: 0.9180
```

![loss](https://user-images.githubusercontent.com/23243761/80826779-14bd2a80-8be3-11ea-8d14-4032585f6664.png)
![val_acc](https://user-images.githubusercontent.com/23243761/80826782-1555c100-8be3-11ea-921e-92c54f06c546.png)

Once we completely trained and tested our model we could get the final result which showed by the following figure
![results](https://user-images.githubusercontent.com/23243761/80826966-6d8cc300-8be3-11ea-87a0-9c0e08cf8aeb.png)

