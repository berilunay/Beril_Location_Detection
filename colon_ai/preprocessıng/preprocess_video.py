import cv2
import numpy as np
import os
from PIL import Image
import pytesseract
import shutil
import csv
from matplotlib import pyplot as plt
from joblib import delayed
from joblib import Parallel


def conv_video2_image():
    fileLoc = '/home/beril/Thesis_Beril/Videos/Video017.mp4'
    cap = cv2.VideoCapture(fileLoc)

    # Create Frames Desired
    FrameSkip = 1
    try:
        if not os.path.exists('/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/Video7'):
            os.makedirs('/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/Video7')
    except OSError:
        print("Error creating directory")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    print(cap.get(cv2.CAP_PROP_FPS))
    current_frame = 0

    while (current_frame < length):
        # Capture Frame - by - Frame
        ret, frame = cap.read()
        name = '/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/Video7/Image' + f'{current_frame:05d}' + '.jpg'

        if (current_frame // FrameSkip == current_frame / FrameSkip):
            print('Creating...' + name)
            cv2.imwrite(name, frame)

        current_frame += 1
    print("--------------------finished creating----------------")
    cap.release()
    cv2.destroyAllWindows()


def create_path_list():
    video_path = "/home/beril/Thesis_Beril/Dataset_preprocess_new/Images"
    temp = []
    for filename in sorted(os.listdir(video_path)):
        im_path = '/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/' + str(filename)
        temp.append(im_path)
    return temp


def trial_labels():
    video_name = "Video7"

    video_path = f"/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/{video_name}"
    filenames = sorted(os.listdir(video_path))

    Parallel(n_jobs=10)(
        delayed(process_image)(video_name=video_name, current=current, filename=filename) for current, filename in
        enumerate(filenames))



def process_image(video_name, current, filename):
    im_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/{video_name}/' + str(filename)
    copy_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Location_Labels/{video_name}/Image' + f'{current:05d}'

    try:
        if not os.path.exists(copy_path):
            os.makedirs(copy_path)

    except OSError:
        print("Error creating directory")
    print(im_path)

    im = Image.open(im_path)
    w, h = im.size
    unit = w // 1.75
    im1 = im.crop((300, 230, 1100, 850))
    file_path = os.path.join(copy_path, "colon" + ".png")
    im1.save(file_path)

    im2 = im.crop((unit, 230, 1910, 850))
    file_path = os.path.join(copy_path, "3D" + ".png")
    im2.save(file_path)


    im3 = im.crop((60, 795, 378, 1072))
    file_path = os.path.join(copy_path, "Quality" + ".png")
    new_path = os.path.join(copy_path, "Quality" + ".txt")
    im3.save(file_path)
    returned_text = detect_empty_image(file_path)
    create_txt_qua(new_path, returned_text)


    im4 = im.crop((unit * 1.4, 847, 1770, 1060))
    file_path = os.path.join(copy_path, "Location" + ".png")
    new_path2 = os.path.join(copy_path, "Location" + ".txt")
    im4.save(file_path)
    returned_text2 = img_txt(file_path)
    create_txt_loc(new_path2, returned_text2)

    if (returned_text2 != "R" and returned_text2 != "L" and returned_text2 != "M"):
        print("There is a problem in image: ", current)
        print("path removed: ", copy_path)
        shutil.rmtree(copy_path)

def new_model_labels():
    video_name = "Video7"

    video_path = f"/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/{video_name}"
    filenames = sorted(os.listdir(video_path))

    Parallel(n_jobs=10)(
        delayed(process_image_newmodel)(video_name=video_name, current=current, filename=filename) for current, filename in
        enumerate(filenames))


def process_image_newmodel(video_name, current, filename):
    im_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/{video_name}/' + str(filename)
    copy_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/procedure_detection/Train_TI_Labels/{video_name}/Image' + f'{current:05d}'

    try:
        if not os.path.exists(copy_path):
            os.makedirs(copy_path)

    except OSError:
        print("Error creating directory")
    print(im_path)

    im = Image.open(im_path)
    w, h = im.size
    unit = w // 1.75
    im1 = im.crop((300, 230, 1100, 850))
    file_path = os.path.join(copy_path, "colon" + ".png")
    im1.save(file_path)

    im2 = im.crop((60, 795, 378, 1072))
    file_path1 = os.path.join(copy_path, "Quality" + ".png")
    label_path = os.path.join(copy_path, "Label" + ".txt")
    im2.save(file_path1)
    returned_text_quality = detect_empty_image(file_path1)

    im4 = im.crop((unit, 230, 1910, 850))
    file_path3 = os.path.join(copy_path, "3D" + ".png")
    im4.save(file_path3)

    im3 = im.crop((unit * 1.4, 30, 1790, 232))#image of TI
    file_path2 = os.path.join(copy_path, "Location" + ".png")
    im3.save(file_path2)
    returned_text_ill = detect_empty_image(file_path2)
    label="N"


    if (returned_text_ill=="TI" and returned_text_quality!="p"):
        create_txt_qua(label_path, returned_text_ill)
    elif (returned_text_ill!="TI" and returned_text_quality!="p"):
        create_txt_qua(label_path, label)
    elif (returned_text_ill!="TI" and returned_text_quality=="p"):
        create_txt_qua(label_path, returned_text_quality)
    else :
        create_txt_qua(label_path, returned_text_ill)


def create_train_label_quality():
    video_name = "Video7"

    video_path = f"/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/{video_name}"
    filenames = sorted(os.listdir(video_path))

    Parallel(n_jobs=10)(
        delayed(process_quality)(video_name=video_name, current=current, filename=filename) for current, filename in
        enumerate(filenames))

def process_quality(video_name, current, filename):

    im_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Images/{video_name}/' + str(filename)
    copy_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Quality_Labels/{video_name}/Image' + f'{current:05d}'

    try:
        if not os.path.exists(copy_path):
            os.makedirs(copy_path)

    except OSError:
        print("Error creating directory")

    print(im_path)
    im = Image.open(im_path)
    w, h = im.size
    unit = w // 1.75

    im1 = im.crop((300, 230, 1100, 850))
    file_path = os.path.join(copy_path, "colon" + ".png")
    im1.save(file_path)

    im3 = im.crop((60, 795, 378, 1072))
    file_path = os.path.join(copy_path, "Quality" + ".png")
    new_path = os.path.join(copy_path, "Quality" + ".txt")
    im3.save(file_path)
    returned_text = detect_empty_image(file_path)
    create_txt_qua(new_path, returned_text)


    if (returned_text != "G" and returned_text != "M" and returned_text != "B"):
        print("The quality is empty: ", current)
        print("path removed: ", copy_path)
        shutil.rmtree(copy_path)



def img_txt(file_path):
    img = cv2.imread(file_path)
    img = cv2.bitwise_not(img)
    custom_config = r'-c tessedit_char_whitelist=BGLMPR --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    split_string = text.split("\n", 1)
    substring = split_string[0]
    print(substring)
    return substring


def create_txt_loc(path, str):
    myText = open(path, 'w')
    myText.write(str)
    myText.close()


def create_txt_qua(path, str2):
    myText2 = open(path, 'w')
    myText2.write(str2)
    myText2.close()


def detect_empty_image(path):
    img_qua = Image.open(path)
    converted_image = np.asarray(img_qua)
    number_of_white_pix = np.sum(converted_image)
    print("sum of pixels: ", number_of_white_pix)
    if number_of_white_pix <= 10000:
        txt_out = "empty"
        print(txt_out)
    else:
        txt_out = img_txt(path)
    return txt_out


def video_label_location():
    video_name = "Video4"

    video_path = f"/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Location_Labels/{video_name}"
    filenames = sorted(os.listdir(video_path))

    Parallel(n_jobs=10)(
        delayed(create_label_folders)(video_name=video_name, count=current, dir_name=filename) for current, filename in
        enumerate(filenames))


def create_label_folders(video_name,count,dir_name):

    copyR = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Location_Labels/location_test/R/{video_name}'
    copyL = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Location_Labels/location_test/L/{video_name}'
    copyM = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Location_Labels/location_test/M/{video_name}'

    print("Started......")

    image_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Location_Labels/{video_name}/' + str(
        dir_name) + "/" + "3D.png"
    img = Image.open(image_path)
    label_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Location_Labels/{video_name}/' + str(
        dir_name) + "/" + "Location.txt"
    loc = open(label_path, 'r')
    string_loc = loc.read()

    if (string_loc == "R"):
        file_path = os.path.join(copyR, "3D_V4_" + f'{count:05d}' + ".png")
        img.save(file_path)
        print(file_path)

    elif (string_loc == "M"):
        file_path = os.path.join(copyM, "3D_V4_" + f'{count:05d}' + ".png")
        img.save(file_path)
        print(file_path)

    else:
        file_path = os.path.join(copyL, "3D_V4_" + f'{count:05d}' + ".png")
        img.save(file_path)
        print(file_path)


def video_label_quality():
    video_name = "Video4"

    video_path = f"/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Quality_Labels/{video_name}"
    filenames = sorted(os.listdir(video_path))

    Parallel(n_jobs=10)(
        delayed(create_video_folder_quality)(video_name=video_name, count=current, dir_name=filename) for current, filename in
        enumerate(filenames))


def create_video_folder_quality(video_name,count,dir_name):

    copyG = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Quality_Labels/quality_train/G/{video_name}'
    copyB = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Quality_Labels/quality_train/B/{video_name}'
    copyM = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Video_Quality_Labels/quality_train/M/{video_name}'

    image_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Quality_Labels/{video_name}/' + str(dir_name) + "/" + "colon.png"
    img = Image.open(image_path)
    label_path = f'/home/beril/Thesis_Beril/Dataset_preprocess_new/Train_Quality_Labels/{video_name}/' + str(dir_name) + "/" + "Quality.txt"
    loc = open(label_path, 'r')
    string_loc = loc.read()

    if (string_loc == "G"):
        file_path = os.path.join(copyG, "3D_V4_" + f'{count:05d}' + ".png")
        img.save(file_path)
        print(file_path)

    elif (string_loc == "B"):
        file_path = os.path.join(copyB, "3D_V4_" + f'{count:05d}' + ".png")
        img.save(file_path)
        print(file_path)

    else:
        file_path = os.path.join(copyM, "3D_V4_" + f'{count:05d}' + ".png")
        img.save(file_path)
        print(file_path)



def clean_P():
    video_name="Video7"
    video_folder_path=f"/home/beril/Thesis_Beril/Train_Labels_Quality/{video_name}"
    for foldername in sorted(os.listdir(video_folder_path)):
        label_path = f'/home/beril/Thesis_Beril/Train_Labels_Quality/{video_name}/' + str(
            foldername) + "/" + "Quality.txt"
        label_folder=f'/home/beril/Thesis_Beril/Train_Labels_Quality/{video_name}/' + str(foldername)
        qua = open(label_path, 'r')
        string_qua = qua.read()
        if (string_qua == "p"):
            print("p detected: ", label_folder)
            shutil.rmtree(label_folder)



if __name__ == '__main__':
    #clean_P()
    #new_model_labels()
    #conv_video2_image()
    #trial_labels()
    #create_train_label_quality()

    video_path="/home/beril/Thesis_Beril/Train_Labels_Quality/Video7"
    filenames = sorted(os.listdir(video_path))
    print("lenght: ",len(filenames))


