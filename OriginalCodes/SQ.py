# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv   #csvモジュールをインポートする
from PIL import Image

def function_perspective():
    def open_with_python_csv(filename):
        csvdata = []
        with open(filename, 'r') as filename:
            reader = csv.reader(filename)
            # ヘッダ行は特別扱い
            csvheader = next(reader)
            # 中身
            for row in reader:
                csvdata.append(row)
        return csvheader, csvdata
    csvheader, csvdata = open_with_python_csv('SQ.csv')

    csvlist = ([['imagename']])
    
    for x in csvdata:
        imagename = x[0]
        LTX = x[1]
        LTY = x[2]
        LBX = x[3]
        LBY = x[4]
        RTX = x[5]
        RTY = x[6]
        RBX = x[7]
        RBY = x[8]

        print(imagename)
    
        orgimg = Image.open(imagename, 'r')
        exif = orgimg._getexif()
#        orientation = exif.get(0x112, 1)
        orientation = 1
        convert_image = {
            # そのまま
            1: lambda img: img,
            # 左右反転
            2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            # 180度回転
            3: lambda img: img.transpose(Image.ROTATE_180),
            # 上下反転
            4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
            # 左右反転＆反時計回りに90度回転
            5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
            # 反時計回りに270度回転
            6: lambda img: img.transpose(Image.ROTATE_270),
            # 左右反転＆反時計回りに270度回転
            7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270), 
            # 反時計回りに90度回転
            8: lambda img: img.transpose(Image.ROTATE_90),
        }
#        thumbnail_img = convert_image[orientation](orgimg)
        conimg = convert_image[orientation](orgimg)
        conname = 'convert_'+imagename
        conimg.save(conname, 'JPEG')
        img = cv2.imread(conname)
        rows,cols,ch = img.shape

        LT = [LTX,LTY] #左上の座標
        LB = [LBX,LBY] #左下の座標
        RT = [RTX,RTY] #右上の座標
        RB = [RBX,RBY] #右下の座標
        
        pts1 = np.float32([LT,LB,RT,RB])
        pts2 = np.float32([[0,0],[0,1000],[1000,0],[1000,1000]])
        
        M = cv2.getPerspectiveTransform(pts1,pts2)
        
        dst = cv2.warpPerspective(img,M,(1000,1000))
        
        plt.subplot(121),plt.imshow(img),plt.title('Input')
        plt.subplot(122),plt.imshow(dst),plt.title('Output')
        plt.show()
        newimagename = 'perspective_'+imagename
        cv2.imwrite(newimagename, dst)

        appendcsvlist=[newimagename]
        csvlist.append(appendcsvlist)
    # ファイルオープン
    f = open('output_SQ.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
                
    # 出力
    writer.writerows(csvlist)
                
    # ファイルクローズ
    f.close()
function_perspective()