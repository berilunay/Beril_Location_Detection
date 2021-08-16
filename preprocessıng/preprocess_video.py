import cv2
import numpy as np
import os
from PIL import Image
import pytesseract
import shutil
import csv


def conv_video2_image():

    fileLoc = '/home/beril/Thesis_Beril/Videos/Video004.mp4'
    cap = cv2.VideoCapture(fileLoc)

    # Create Frames Desired
    FrameSkip = 25
    try:
        if not os.path.exists('/home/beril/Thesis_Beril/Images'):
            os.makedirs('/home/beril/Thesis_Beril/Images')
        if not os.path.exists('/home/beril/Thesis_Beril/Images/Video3'):
            os.makedirs('/home/beril/Thesis_Beril/Images/Video3')
    except OSError:
        print("Error creating directory")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    print(cap.get(cv2.CAP_PROP_FPS))
    current_frame = 0

    while (current_frame < length):
        # Capture Frame - by - Frame
        ret, frame = cap.read()
        name = '/home/beril/Thesis_Beril/Images/Video3/Image'+f'{current_frame:05d}'+'.jpg'

        if (current_frame // FrameSkip == current_frame / FrameSkip):
            print('Creating...' + name)
            cv2.imwrite(name,frame)

        current_frame += 1
    print("--------------------finished creating----------------")
    cap.release()
    cv2.destroyAllWindows()


def test_method():

    video_path="D:\\Beril\\Thesis\\Data\\TestFolder\\Images"
    current = 1
    for filename in os.listdir(video_path):

        im_path="D:\\Beril\\Thesis\\Data\\TestFolder\\Images\\" +str(filename)
        copy_path="D:\\Beril\\Thesis\\Data\\TestFolder\\labels\\Image"+ f'{current:05d}'

        try:
            if not os.path.exists(copy_path):
                os.makedirs(copy_path)

        except OSError:
            print("Error creating directory")
        print(im_path)

        im = Image.open(im_path)
        # w, h = im.size
        # unit = w // 1.75

        # im1 = im.crop((300, 230, 1100, 850))
        # file_path = os.path.join(copy_path, "colon" + ".png")
        # im1.save(file_path)
        #
        # im2 = im.crop((unit, 230, 1910, 850))
        # file_path = os.path.join(copy_path, "3D" + ".png")
        # im2.save(file_path)

        # image_p="D:\\Beril\\Thesis\\Data\\TestFolder\\Images\\Image12180.jpg"
        # im = Image.open(image_p)
        # w, h = im.size
        # unit = w // 1.75


        # im3 = im.crop((60, 795, 378, 1072))
        # file_path = os.path.join(copy_path, "Quality" + ".png")
        # new_path = os.path.join(copy_path, "Quality" + ".txt")
        # im3.save(file_path)
        # returned_text = detect_empty_image(file_path)
        # create_txt_qua(new_path, returned_text)


        # im4 = im.crop((unit * 1.4, 780, 1770, 1020))
        # im4 = im.crop((unit * 1.4, 847, 1770, 1060))
        # file_path = os.path.join(copy_path, "Location" + ".png")
        # new_path2 = os.path.join(copy_path, "Location" + ".txt")
        # im4.save(file_path)
        # returned_text2 = img_txt(file_path)
        # create_txt_loc(new_path2, returned_text2)
        # if (returned_text2 != "R" and returned_text2 != "L" and returned_text2 != "M"):
        #     print("There is a problem in image: ", current)
        # current += 1


def trial_labels():

    video_path="/home/beril/Thesis_Beril/Images/Video3"
    current=1
    for filename in sorted(os.listdir(video_path)):
        im_path = '/home/beril/Thesis_Beril/Images/Video3/' + str(filename)
        copy_path='/home/beril/Thesis_Beril/Train_Labels/Video3/Image'+f'{current:05d}'

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

        #im3 = im.crop((90,780, 400,1020))
        #im3 = im.crop((60,790,388,1060))
        im3 = im.crop((60, 795, 378, 1072))
        file_path = os.path.join(copy_path, "Quality" + ".png")
        new_path = os.path.join(copy_path, "Quality" + ".txt")
        im3.save(file_path)
        returned_text= detect_empty_image(file_path)
        create_txt_qua(new_path,returned_text)

        #im4 = im.crop((unit * 1.4, 780, 1770, 1020))
        im4 = im.crop((unit * 1.4, 847, 1770, 1060))
        file_path = os.path.join(copy_path, "Location" + ".png")
        new_path2 = os.path.join(copy_path, "Location" + ".txt")
        im4.save(file_path)
        returned_text2 = img_txt(file_path)
        create_txt_loc(new_path2, returned_text2)
        if( returned_text2!= "R" and  returned_text2!= "L" and returned_text2!= "M" ):
            print("There is a problem in image: ",current)
            print("path removed: ",copy_path)
            shutil.rmtree(copy_path)
            current = current - 1
        current+=1

def video_to_image_quality():
    fileLoc = '/home/beril/Thesis_Beril/Videos/Video006.mp4'
    cap = cv2.VideoCapture(fileLoc)

    # Create Frames Desired
    FrameSkip = 25
    try:
        if not os.path.exists('/home/beril/Thesis_Beril/Images_Quality'):
            os.makedirs('/home/beril/Thesis_Beril/Images_Quality')
        if not os.path.exists('/home/beril/Thesis_Beril/Images_Quality/Video4'):
            os.makedirs('/home/beril/Thesis_Beril/Images_Quality/Video4')
    except OSError:
        print("Error creating directory")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    print(cap.get(cv2.CAP_PROP_FPS))
    current_frame = 0

    while (current_frame < length):
        # Capture Frame - by - Frame
        ret, frame = cap.read()
        name = '/home/beril/Thesis_Beril/Images_Quality/Video4/Image' + f'{current_frame:05d}' + '.jpg'

        if (current_frame // FrameSkip == current_frame / FrameSkip):
            print('Creating...' + name)
            cv2.imwrite(name, frame)

        current_frame += 1
    print("--------------------finished creating----------------")
    cap.release()
    cv2.destroyAllWindows()

def create_train_label_quality():
    video_path = "/home/beril/Thesis_Beril/Images_Quality/Video4"
    current = 1
    for filename in sorted(os.listdir(video_path)):
        im_path = '/home/beril/Thesis_Beril/Images_Quality/Video4/' + str(filename)
        copy_path = '/home/beril/Thesis_Beril/Train_Labels_Quality/Video4/Image' + f'{current:05d}'

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
        #part to skip the empty frames
        #if (returned_text == "empty"):
        if (returned_text != "G" and returned_text != "p" and returned_text != "M" and returned_text != "B"):
            print("The quality is empty: ", current)
            print("path removed: ", copy_path)
            shutil.rmtree(copy_path)
            current=current-1
        current += 1
    print("Total Quality Labels: ", current)



def img_txt(file_path):
    img = cv2.imread(file_path)
    img = cv2.bitwise_not(img)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #pytesseract.pytesseract.tesseract_cmd = r'/usr/share/tesseract-ocr/4.00/tessdata'
    custom_config = r'-c tessedit_char_whitelist=BGLMPR --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    split_string = text.split("\n", 1)
    substring = split_string[0]
    print(substring)
    return substring

def create_txt_loc(path,str):
    myText = open(path, 'w')
    myText.write(str)
    myText.close()

def create_txt_qua(path,str2):
    myText2 = open(path, 'w')
    myText2.write(str2)
    myText2.close()

def detect_empty_image(path):
    img_qua= Image.open(path)
    converted_image = np.asarray(img_qua)
    #print(converted_image)
    #number_of_white_pix = np.sum(converted_image == 255)
    number_of_white_pix = np.sum(converted_image)
    print("sum of pixels: ",number_of_white_pix)
    if number_of_white_pix<=10000:
        txt_out = "empty"
        print(txt_out)
    else:
        txt_out = img_txt(path)
    return txt_out


def create_label_folders():
    main_path='/home/beril/Thesis_Beril/Train_Labels/Video5'
    copyR ='/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/R'
    copyL ='/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/L'
    copyM ='/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/M'
    count_R=1
    count_L=1
    count_M=1
    print("Started......")
    for dir_name in os.listdir(main_path):
            image_path='/home/beril/Thesis_Beril/Train_Labels/Video5/'+ str(dir_name) + "/"+ "3D.png"
            img = Image.open(image_path)
            label_path = '/home/beril/Thesis_Beril/Train_Labels/Video5/' + str(dir_name) + "/" + "Location.txt"
            loc = open(label_path, 'r')
            string_loc = loc.read()


            if (string_loc=="R"):
                file_path = os.path.join(copyR, "3D_V5_" +f'{count_R:05d}'+ ".png")
                img.save(file_path)
                count_R = count_R + 1

            elif(string_loc=="M"):
                file_path = os.path.join(copyM, "3D_V5_" +f'{count_M:05d}'+ ".png")
                img.save(file_path)
                count_M = count_M + 1

            else:
                file_path = os.path.join(copyL, "3D_V5_" + f'{count_L:05d}' + ".png")
                img.save(file_path)
                count_L = count_L + 1


    print("Images in R: ",count_R)
    print("Images in L: ", count_L)
    print("Images in M: ", count_M)


def create_csv():
    f = open('/home/beril/Thesis_Beril/Dataset_VideoCNN/Dataset_VideoCNN_Paths.csv', 'w')
    # path1=[[/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/L]]
    # path2=['/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/M']
    # path3=['/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/R']
    # create the csv writer
    path_list=['/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/R','/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/M','/home/beril/Thesis_Beril/Dataset_VideoCNN/Train_Location_Video/L']
    writer = csv.writer(f)


    # write a row to the csv file
    for path in path_list:
        writer.writerow([str(path)])


    # close the file
    f.close()


if __name__ == '__main__':
    #trial_labels()
    #video_to_image_quality()
    #create_train_label_quality()
    #create_label_folders()
    create_csv()


















