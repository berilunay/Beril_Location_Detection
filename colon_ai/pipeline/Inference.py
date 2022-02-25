import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from cv2 import imshow
from matplotlib import pyplot as plt
from numpy import shape
from pytorch_lightning import Trainer
from torchvision.datasets import VisionDataset
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from colon_ai.TI_model.DatamodelTI import ColonModule_TI
from colon_ai.pipeline.DatasetLocation_Inference import InferenceDatasetLocation
from colon_ai.train_location.DataLoader_Location import ColonDataModuleLocation
from colon_ai.train_location.DataModelLocation import ColonModuleLocation
from colon_ai.train_location.DatasetClass_Location import ColonDatasetLocation
from colon_ai.traınıng.DataModelColon import ColonModuleQuality
import itertools

class InferenceDatasetQuality(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, num_samples=None):
        super(InferenceDatasetQuality, self).__init__(root, transform=transform, target_transform=target_transform)
        self.video_dirs = []
        self.video_dirs.append(self.root)
        self.sample_dirs = []  # image0,image1 ...
        for video_dir in self.video_dirs:
            self.sample_dirs += (sorted(Path(video_dir).iterdir())) # gets image and labels(folder)
        if num_samples:
            self.sample_dirs = self.sample_dirs[:num_samples]


    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]
        colon= self._load_and_transform_colon(sample_dir)

        return colon


    def _load_and_transform_colon(self, sample_dir):
        colon = self._load_colon(sample_dir)
        colon = self._apply_transforms(colon)

        return colon


    def _load_colon(self, sample_dir):
        im_path = str(sample_dir / "colon.png")
        colon=cv2.imread(im_path)
        resized_image = cv2.resize(colon, (224, 224),interpolation=cv2.INTER_NEAREST)
        img_new=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        colon = np.array(img_new)
        colon=np.moveaxis(colon,-1,0)
        colon = torch.from_numpy(colon)
        colon= colon/255.0

        return colon


    def _apply_transforms(self, colon):
        if self.transform is not None:
            colon = self.transform(colon)

        return colon


    def __len__(self):
        return len(self.sample_dirs)


"""Gets the predicted output labels of the TI model"""
def show_ouput_TI(model, dataloader, class_dict=None):
    TI_labels=[]
    for features in dataloader:
        with torch.no_grad():
            logits = model(features)
            predictions = torch.argmax(logits,dim=1)
            pred_numpy=predictions.numpy()
            TI_labels.append(pred_numpy)


    pred_labels_conv = np.concatenate(TI_labels, axis=None)
    print("predicted labels conv TI: ", pred_labels_conv)
    converted_arr=[]

    for num in pred_labels_conv:
        converted_arr.append(class_dict[num])
    print("output array TI: ",converted_arr)
    print("len output array TI: ",len(converted_arr))


    return converted_arr

"""Prints the average quality for every 5 frames and visualize it with grid"""
def show_ouput_quality(model, dataloader,start_index, class_dict=None):
    quality_labels=[]
    dict_map = {'G':5,'M':3,'B':1}
    for features in dataloader:
        with torch.no_grad():
            logits = model(features)
            predictions = torch.argmax(logits,dim=1)
            pred_numpy=predictions.numpy()
            quality_labels.append(pred_numpy)

    pred_labels_conv = np.concatenate(quality_labels, axis=None)
    print("predicted labels conv quality: ", pred_labels_conv)
    #coversion
    out_label_quality=[] #actual output labels coming from model
    count=0
    result=""
    output_avg_numeric=[] #avg of the each 5 elements as numeric value
    out_label_avg_qua=[] #corresponding average quality label
    calc=0

    for num in pred_labels_conv:
        out_label_quality.append(class_dict[num])
    print("output array quality: ", out_label_quality)
    print("len out array quality: ", len(out_label_quality))
    index = start_index

    """calculation part,1 sec has 25 frames, to print avg every 5 sec count must be 125"""
    for i in range(start_index,len(out_label_quality)):
        calc=calc+dict_map[out_label_quality[i]]
        count+=1
        index=index+1
        if count==125:
            avg = calc /125
            if avg < 2.0:
                result = "bad"
            elif avg >= 2.0 and avg < 4.0:
                result = "middle"
            else:
                result = "good"
            output_avg_numeric.append(avg)
            out_label_avg_qua.append(result)
            print("average quality "+"between frames "+str(index-125)+"-"+str(index)+ " is:",result)
            count=0
            calc=0

    """checking the avg quality labels for every 5 frame"""
    print("output_label_avg: ", out_label_avg_qua)
    print("output_label_avg length: ", len(out_label_avg_qua))

    """Adding the last remaining elements"""
    sum_elem=0
    remained_elem= len(out_label_quality) % 125
    if remained_elem!=0:
        for k in range(remained_elem):
            sum_elem = sum_elem + dict_map[out_label_quality[k - remained_elem]]
        avg_add=sum_elem/remained_elem
        if avg_add< 2.0:
            result = "bad"
        elif avg_add >= 2.0 and avg_add < 4.0:
            result = "middle"
        else:
            result = "good"
        output_avg_numeric.append(avg_add)
        out_label_avg_qua.append(result)
        print("average quality " + "between frames " + str(len(out_label_quality) - remained_elem) + "-" + str(len(out_label_quality)) + " is:", result)
        print("avg output quality numeric: ",output_avg_numeric)
        print("avg output quality with label: ", out_label_avg_qua)


    return out_label_quality, out_label_avg_qua, output_avg_numeric

"""Returns the predicted location labels"""
def show_ouput_location(model, dataloader, class_dict=None):
    location_labels=[]

    for features in dataloader:
        with torch.no_grad():
            logits = model(features)
            predictions = torch.argmax(logits,dim=1)
            pred_numpy=predictions.numpy()
            location_labels.append(pred_numpy)

    pred_labels_conv = np.concatenate(location_labels, axis=None)
    print("predicted labels conv location: ", pred_labels_conv)

    converted_arr=[]
    for num in pred_labels_conv:
         converted_arr.append(class_dict[num])
    print("out array location: ",converted_arr)

    return converted_arr

"""Puts predicted location text onto corresponding scope image and saves it"""
def show_images_with_labels(dataloader, output_array_loc,start_index,output_arr_TI):
    print(".....Location frame preprocesing started.........")

    save_path="/home/beril/Thesis_Beril/Inference/Frames_Loc/Video6"
    index=0
    font_path = "/home/beril/Thesis_Beril/Inference/Font/Swansea-q3pd.ttf"
    font_chs = ImageFont.truetype(font_path, 20)

    for images in dataloader:
        input_img = images
        c, t, h, w = input_img.shape
        input_img = torch.permute(input_img,(0, 2, 3, 1))
        input_img_np=input_img.numpy()

    for i in range(len(input_img)):
        img=input_img_np[i]
        im = Image.fromarray((img*255).astype('uint8'), 'RGB')
        TI_label=output_arr_TI[i]
        I1 = ImageDraw.Draw(im)
        I1.text((173, 20), str(TI_label), fill=(150, 48, 27), font=font_chs)
        if(i>=start_index):
            label = output_array_loc[i]
            I2 = ImageDraw.Draw(im)
            I2.text((22, 37), "Location: "+str(label), fill=(150, 48, 27),font=font_chs)
        file_path = os.path.join(save_path, "Image" + f'{index:05d}' + ".png")
        im.save(file_path)
        index=index+1


"""Puts predicted quality text onto corresponding colon image and saves it"""
def show_images_with_labels_quality(dataloader,output_array,out_arrayTI,start_index):
    print(".....Quality frame preprocesing started.........")

    save_path="/home/beril/Thesis_Beril/Inference/Frames_Quality/Video6"
    index=0
    font_path="/home/beril/Thesis_Beril/Inference/Font/Swansea-q3pd.ttf"
    font_chs = ImageFont.truetype(font_path, 20)

    for images in dataloader:
        input_img = images
        c, t, h, w = input_img.shape
        input_img = torch.permute(input_img,(0, 2, 3, 1))
        input_img_np=input_img.numpy()

    for i in range(len(input_img)):
        img=input_img_np[i]
        im = Image.fromarray((img*255).astype('uint8'), 'RGB')
        label=output_array[i]
        if(i>=start_index):
            if(out_arrayTI[i]=="p"):
                I1 = ImageDraw.Draw(im)
                I1.text((28, 37), "Procedure", fill=(9, 28, 173),font=font_chs)
            else:
                I2 = ImageDraw.Draw(im)
                I2.text((28, 37), "Quality:" + str(label), fill=(9, 28, 173), font=font_chs)
            file_path = os.path.join(save_path, "Image" + f'{index:05d}' + ".png")
            im.save(file_path)
            index=index+1
        else:
            file_path = os.path.join(save_path, "Image" + f'{index:05d}' + ".png")
            im.save(file_path)
            index = index + 1

"""Concatanates scope and colon view into one image"""
def concat_images():
    print("............Concat Location and Quality Started............")
    main_path="/home/beril/Thesis_Beril/Inference/Frames_Loc/Video6"
    copy_path = "/home/beril/Thesis_Beril/Inference/Frames_Concat/Video6"

    count=0
    for file_name in sorted(os.listdir(main_path)):
        im1_path = '/home/beril/Thesis_Beril/Inference/AVG_Quality_Concat/Video6/'+str(file_name)
        img1 = Image.open(im1_path)
        im2_path='/home/beril/Thesis_Beril/Inference/Frames_Loc/Video6/' + str(file_name)
        img2 = Image.open(im2_path)
        print(im2_path)
        new_img = Image.new('RGB', (img1.width + img2.width, img1.height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
        file_path = os.path.join(copy_path, "Image_Concat" + f'{count:05d}' + ".png")
        new_img.save(file_path)
        count = count+ 1

"""Concatanates avg quality frame with the quality frame"""
def concat_quality_and_avg_quality(avg_quality_out_arr,start_index):
    print(".........concataneting quality frames................")
    quality_frame_path="/home/beril/Thesis_Beril/Inference/Frames_Quality/Video6"
    save_concat_path="/home/beril/Thesis_Beril/Inference/AVG_Quality_Concat/Video6"
    count=0
    length_array=0
    font_path = "/home/beril/Thesis_Beril/Inference/Font/Swansea-q3pd.ttf"
    font_chs = ImageFont.truetype(font_path, 18)
    index=0

    for file_name in sorted(os.listdir(quality_frame_path)):
        im1_path = '/home/beril/Thesis_Beril/Inference/Frames_Quality/Video6/' + str(file_name)
        img1 = Image.open(im1_path)
        img2 = Image.new('RGB', (img1.width, 50), color=(172, 184, 191))
        new_img1 = Image.new('RGB', (img1.width, img1.height + img2.height))
        new_img1.paste(img1, (0, 0))
        new_img1.paste(img2, (0, img1.height))

        if(count>=start_index):
            if(count!=start_index and count%125==0):
                I1 = ImageDraw.Draw(img2)
                I1.text((13, 20), "Average quality is: " + str(avg_quality_out_arr[length_array]), fill=(25, 25, 26),font=font_chs)
                new_img = Image.new('RGB', (img1.width, img1.height + img2.height))
                new_img.paste(img1, (0, 0))
                new_img.paste(img2, (0, img1.height))
                file_path = os.path.join(save_concat_path, "Image" + f'{count:05d}' + ".png")
                new_img.save(file_path)
                length_array=length_array+1
            else:
                if(count>(125+start_index)):
                    I2 = ImageDraw.Draw(img2)
                    I2.text((13, 20), "Average quality is: " + str(avg_quality_out_arr[length_array-1]), fill=(25, 25, 26),font=font_chs)
                    new_img2 = Image.new('RGB', (img1.width, img1.height + img2.height))
                    new_img2.paste(img1, (0, 0))
                    new_img2.paste(img2, (0, img1.height))
                    file_path = os.path.join(save_concat_path, "Image" + f'{count:05d}' + ".png")
                    new_img2.save(file_path)
                else:
                    file_path_org = os.path.join(save_concat_path, "Image" + f'{count:05d}' + ".png")
                    new_img1.save(file_path_org)

        else:
            file_path_org = os.path.join(save_concat_path, "Image" + f'{count:05d}' + ".png")
            new_img1.save(file_path_org)

        count = count + 1


"""Creates video from image frames"""
def image_to_video():
    print("converting to video...")
    image_folder = "/home/beril/Thesis_Beril/Inference/Frames_Concat/Video6"
    video_name = "/home/beril/Thesis_Beril/Inference/Video_output/video6(23).mp4"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video = cv2.VideoWriter(video_name, fourcc, 25, (width, height))

    for image in sorted(images):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def longest_20TI_index(output_arr_TI):
    out_test=["TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI","TI"]
    test_list=[]
    index=0
    for i in range(len(output_arr_TI)-20):
        if(output_arr_TI[i:(i+20)]==out_test):
             test_list.append(i+19)
    #new addition if test list is empty
    if(len(test_list)==0):
        for i in range(len(output_arr_TI)):
            if(output_arr_TI[i]=="TI"):
                test_list.append(i)
                index = test_list[-1] + 1
    else:
        print("test_list:",test_list)
        print("len test_list:",len(test_list))
        middle=len(test_list)//2
        index=test_list[middle]+1
        print("index: ",index)
    return index

def calculate_average(output_arr_quality):
    sum_avg= sum(output_arr_quality)
    avg_out=sum_avg/ len(output_arr_quality)
    print("avg_output: ",avg_out)
    if avg_out < 2.0:
        result = "bad"
    elif avg_out >= 2.0 and avg_out < 4.0:
        result = "middle"
    else:
        result = "good"
    print("avg result is: ",result)
    return result

def create_summary_report(output_arr_location,output_arr_qua,start_index):
    path="/home/beril/Thesis_Beril/Inference/Summary/video6_summary_23.txt"
    countR=0
    countM=0
    countL=0
    dict_map = {'G': 5, 'M': 3, 'B': 1}
    quality_listR=[]
    quality_listM=[]
    quality_listL=[]

    for i in range(start_index,len(output_arr_qua)):
        if(output_arr_location[i]=="R"):
            countR=countR+1
            quality_listR.append(dict_map[output_arr_qua[i]])
        elif(output_arr_location[i]=="M"):
            countM = countM + 1
            quality_listM.append(dict_map[output_arr_qua[i]])
        else:
            countL = countL + 1
            quality_listL.append(dict_map[output_arr_qua[i]])

    time_spent_R= countR/25
    time_spent_L =countL/25
    time_spent_M =countM/25

    total_time= (time_spent_R+ time_spent_L+ time_spent_M)/60.0
    print("time spent in R(sec): ",time_spent_R)
    print("time spent in M(sec): ", time_spent_M)
    print("time spent in L(sec): ", time_spent_L)


    avg_qualityR =calculate_average(quality_listR)
    avg_qualityM =calculate_average(quality_listM)
    avg_qualityL =calculate_average(quality_listL)


    strR_time="Time spent in R is: "+str(time_spent_R)+" seconds\n"
    strR_avg = "Average quality in R is: " + str(avg_qualityR) + "\n"
    strM_time="Time spent in M is: " + str(time_spent_M) + " seconds\n"
    strM_avg ="Average quality in M is: " + str(avg_qualityM) + "\n"
    strL_time="Time spent in L is: " + str(time_spent_L) + " seconds\n"
    strL_avg=" Average quality in L is: " + str(avg_qualityL) + "\n"
    str_total=" Total colonoscopy time after TI is: " + str(total_time) + "\n"

    #writing the results to the txt file
    print_list=[strR_time,strR_avg,strM_time,strM_avg,strL_time,strL_avg,str_total]
    myText = open(path, 'w')
    myText.writelines(print_list)
    myText.close()

    time_out=[time_spent_R,time_spent_M,time_spent_L]

    """additional for printing the time in minutes
    #time_spentR_min= time_spent_R /60.0
    #time_spentM_min= time_spent_M /60.0
    #time_spentL_min= time_spent_L /60.0"""

    return time_out

def plot_report(avg_output_quality,time_out):
    y_out=avg_output_quality
    time_spent_each=time_out
    end_sec=(len(y_out)*5)
    barWidth = 0.25
    # make data
    x = np.arange(5, end_sec+5, 5)
    x_show=np.arange(5, end_sec+5, 10)
    x_bar=["R","M","L"]

    fig, ax = plt.subplots(figsize=(20, 6))
    # # Define x and y axes
    ax.plot(x,y_out, marker = 'o',color = 'darkblue', alpha = 0.3)
    plt.xticks(x_show)
    # Set plot title and axes labels
    ax.set(title="Summary Plot of The Procedure after TI",
           xlabel="time(sec)",
           ylabel="Average Quality")

    fig2, ax2 = plt.subplots(figsize=(20, 6))
    plt.bar(x_bar, time_spent_each, color =['r','b','g'], width = barWidth,
        edgecolor ='grey', label ='IT')
    ax2.set(title="Time Spent in Each Location",
               xlabel="Location",
               ylabel="Time Spent")

    plt.show()


if __name__ == '__main__':

    print("Code is running...")
    Test_Path="/home/beril/Thesis_Beril/Dataset_preprocess_new/procedure_detection/Test_TI_Labels/Video6"
    val_test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    #TI Model Outputs
    TI_dataset = InferenceDatasetQuality(root=Test_Path,transform=val_test_transform)
    TI_dict = {0: 'TI', 1: 'p', 2: 'N'}
    checkpoint_model_path_TI="/home/beril/BerilCodes/ColonAI_LocationDetection/colon_ai/TI_model/uncategorized/best_model/checkpoints/besthparamstd--epoch=2-val_loss=0.35-val_acc=0.93--train_loss=0.22-train_acc=0.95.ckpt"
    pretrained_model_TI = ColonModule_TI.load_from_checkpoint(checkpoint_path=checkpoint_model_path_TI)
    pretrained_model_TI.eval()
    TI_loader = DataLoader(TI_dataset, batch_size=pretrained_model_TI.hparams["batch_size"], num_workers=4)
    out_arr_TI=show_ouput_TI(pretrained_model_TI, TI_loader, TI_dict)
    index_start_TI=longest_20TI_index(out_arr_TI)
    print("index start: ",index_start_TI)


    #Location Detection Model Outputs
    location_dataset = InferenceDatasetLocation(root=Test_Path,transform=val_test_transform)
    location_dict = {0: 'R', 1: 'M', 2: 'L'}
    checkpoint_model_path_loc="/home/beril/BerilCodes/ColonAI_LocationDetection/colon_ai/train_location/uncategorized/best_model(11.12)/checkpoints/besthparamstd--epoch=5-val_loss=1.11-val_acc=0.56-train_loss=0.01-train_acc=1.00-F1_val=0.59-F1_train=1.00.ckpt"
    pretrained_model_loc = ColonModuleLocation.load_from_checkpoint(checkpoint_path=checkpoint_model_path_loc)
    pretrained_model_loc.eval()
    location_loader = DataLoader(location_dataset, batch_size=pretrained_model_loc.hparams["batch_size"], num_workers=4)
    out_arr_loc=show_ouput_location(pretrained_model_loc, location_loader, location_dict)
    """Second dataloader is needed in order get the original image frames for the views rather than the normalized image views"""
    location_dataset2 = InferenceDatasetLocation(root=Test_Path)
    location_loader2 = DataLoader(location_dataset2, batch_size=len(location_dataset2))
    show_images_with_labels(location_loader2,out_arr_loc,index_start_TI,out_arr_TI)



    #Quality Detection Model Outputs
    quality_dict = {0: 'G', 1: 'M', 2: 'B'}
    quality_dataset = InferenceDatasetQuality(root=Test_Path,transform=val_test_transform)
    checkpoint_model_path = "/home/beril/BerilCodes/ColonAI_LocationDetection/colon_ai/traınıng/uncategorized/bestmodel(11.12)/checkpoints/besthparamstd--epoch=0-val_loss=0.78-val_acc=0.61--train_loss=0.63-train_acc=0.75--F1_train=0.76-F1_val=0.67.ckpt"
    pretrained_model = ColonModuleQuality.load_from_checkpoint(checkpoint_path= checkpoint_model_path)
    pretrained_model.eval()
    quality_loader = DataLoader(quality_dataset, batch_size=pretrained_model.hparams["batch_size"], num_workers=4)
    output_arr_quality,output_avg_qua_label, output_avg_quality_numeric=show_ouput_quality(pretrained_model, quality_loader,index_start_TI, quality_dict)
    quality_dataset2 = InferenceDatasetQuality(root=Test_Path)
    quality_loader2 = DataLoader(quality_dataset2, batch_size=len(quality_dataset2))
    show_images_with_labels_quality(quality_loader2,output_arr_quality,out_arr_TI,index_start_TI)
    concat_quality_and_avg_quality(output_avg_qua_label,index_start_TI)


    #Creating the output video
    concat_images()
    image_to_video()

    #create report and plot
    time_out_arr= create_summary_report(out_arr_loc, output_arr_quality, index_start_TI)
    plot_report(output_avg_quality_numeric, time_out_arr)



