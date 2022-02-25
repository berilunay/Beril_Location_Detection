# Beril_Location_Detection

This project describes a pipeline which uses CNNs to detect location and quality in the colonoscopy. The model is trained with real life colonoscopy videos. As a result, an output colonoscopy
video which has average quality results for every five seconds together with predictedlocation and quality labels is constructed.

The consists of 7 different folders where each one is responsible fora seperate task. The general information regarding the folders are explained as follows:

TI_modeL: The code for TI, procedure and N labels using CNN. The input image is colon view.

Training: The code for detection quality labels (Good(G), Bad(B) and Midde(M)) using CNN. The input image is colon view.

Train_Location: The code for detection location labels(Right(R), Middle(M) and Left(L)) using CNN. The input image is scope view.

Video_CNN: The 3D CNN architecture implemented using PytorchVideo to detect the quality labels(G,M,B).

VideCNN_Location: The 3D CNN architecture implemented using PytorchVideo to detect the location labels(R,M,L).

Preprocessing: Date preprocessing pipeline implementation. Consists of different functions which allows the user to prepare the dataset for each classification model seperately.
It starts with video to image conversion where each image frame is extracted from the video according to its frame rate. Then each image frame is cropped and so that labels, colon
view and the scope view is seperated from each other. Strings on the image frames are extracted as txt using the Tessearct library. Following this each image is saved with their labels
into the corresponding folders that are created for each classification model.

Pipeline: This folder contains the code for the output video creation after the training is done. The predicted labels that are obtained using the most promising models inserted 
on the image frames. Average quality calculation takes place inside this code. At the end of the code the output video is recreated with the predicted TI,procedure,location and quality
labels on it and average quality is printed on the video for every 5 seconds.  Output video is supported with a summary report and average quality summary plots at the end.
