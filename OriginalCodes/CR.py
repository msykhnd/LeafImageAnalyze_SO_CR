# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import csv  # csvモジュールをインポートする
from PIL import Image
from scipy import signal


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

    csvheader, csvdata = open_with_python_csv('CR.csv')

    csvlist = ([['imagename', 'area(px)', 'area(cm)']])

    # 画像ごとに処理
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
        CR1_1_org = x[9]
        CR1_2_org = x[10]
        CR2_1_org = x[11]
        CR2_2_org = x[12]
        CR3_1_org = x[13]
        CR3_2_org = x[14]
        CR4_1_org = x[15]
        CR4_2_org = x[16]

        print(imagename)
        # 画像の読み込み（向きを設定）
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
        conname = imagename + '_convert.jpg'
        conimg.save(conname, 'JPEG')
        img = cv2.imread(conname)
        rows, cols, ch = img.shape

        # 透視変形
        LT = [LTX, LTY]  # 左上の座標
        LB = [LBX, LBY]  # 左下の座標
        RT = [RTX, RTY]  # 右上の座標
        RB = [RBX, RBY]  # 右下の座標

        pts1 = np.float32([LT, LB, RT, RB])
        pts2 = np.float32([[200, 200], [200, 1200], [1200, 200], [1200, 1200]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(img, M, (1400, 1400))

        plt.subplot(121), plt.imshow(img), plt.title('Input')
        plt.subplot(122), plt.imshow(dst), plt.title('Output')
        plt.show()
        cv2.imwrite('perspective_' + imagename, dst)

        # 透視変形に伴う円座標の変換
        CR1 = np.float32([CR1_1_org, CR1_2_org])  # 円の座標その１
        CR2 = np.float32([CR2_1_org, CR2_2_org])  # 円の座標その２
        CR3 = np.float32([CR3_1_org, CR3_2_org])  # 円の座標その３
        CR4 = np.float32([CR4_1_org, CR4_2_org])  # 円の座標その４

        CR1_2 = np.append(CR1, np.float32([1]))
        CR1_per = M.dot(CR1_2)
        CR1_per_2 = (CR1_per[0] / CR1_per[2], CR1_per[1] / CR1_per[2])
        # print (CR1_per_2)

        CR2_2 = np.append(CR2, np.float32([1]))
        CR2_per = M.dot(CR2_2)
        CR2_per_2 = (CR2_per[0] / CR2_per[2], CR2_per[1] / CR2_per[2])
        # print (CR2_per_2)

        CR3_2 = np.append(CR3, np.float32([1]))
        CR3_per = M.dot(CR3_2)
        CR3_per_2 = (CR3_per[0] / CR3_per[2], CR3_per[1] / CR3_per[2])
        # print (CR3_per_2)

        CR4_2 = np.append(CR4, np.float32([1]))
        CR4_per = M.dot(CR4_2)
        CR4_per_2 = (CR4_per[0] / CR4_per[2], CR4_per[1] / CR4_per[2])
        # print (CR4_per_2)

        # @brief 最小二乗法による円フィッティングモジュール

        CR1 = CR1_per_2
        CR2 = CR2_per_2
        CR3 = CR3_per_2
        CR4 = CR4_per_2

        def CircleFitting(x, y):
            """最小二乗法による円フィッティングをする関数
                input: x,y 円フィッティングする点群
        
                output  cxe 中心x座標
                        cye 中心y座標
                        re  半径
        
                参考
                一般式による最小二乗法（円の最小二乗法）　画像処理ソリューション
                http://imagingsolution.blog107.fc2.com/blog-entry-16.html
            """

            sumx = sum(x)
            sumy = sum(y)
            sumx2 = sum([ix ** 2 for ix in x])
            sumy2 = sum([iy ** 2 for iy in y])
            sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

            F = np.array([[sumx2, sumxy, sumx],
                          [sumxy, sumy2, sumy],
                          [sumx, sumy, len(x)]])

            G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                          [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                          [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

            T = np.linalg.inv(F).dot(G)

            cxe = float(T[0] / -2)
            cye = float(T[1] / -2)
            re = math.sqrt(cxe ** 2 + cye ** 2 - T[2])
            # print (cxe,cye,re)
            return (cxe, cye, re)

        if __name__ == '__main__':
            """Unit Test"""
            # 推定する円パラメータ
            # cx=4;   #中心x
            # cy=10;  #中心y
            # r=30;   #半径

            # 円の点群の擬似情報
            plt.figure()
            x = [CR1[0], CR2[0], CR3[0], CR4[0]]
            y = [CR1[1], CR2[1], CR3[1], CR4[1]]

            # x=range(-10,10);
            # y=[]
            # for xt in x:
            #    y.append(cy+math.sqrt(r**2-(xt-cx)**2))

            # 円フィッティング
            (cxe, cye, re) = CircleFitting(x, y)

            # 円描画
            theta = np.arange(0, 2 * math.pi, 0.1)
            xe = []
            ye = []
            for itheta in theta:
                xe.append(re * math.cos(itheta) + cxe)
                ye.append(re * math.sin(itheta) + cye)
            xe.append(xe[0])
            ye.append(ye[0])

            cx = cxe;  # 中心x
            cy = cye;  # 中心y
            r = re;  # 半径

            plt.plot(x, y, "ob", label="raw data")
            plt.plot(xe, ye, "-r", label="estimated")
            plt.plot(cx, cy, "xb", label="center")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.show()
            print(cx, cy, r)
            cx = round(cx)
            cy = round(cy)
            r = round(r)
            print(cx, cy, r)

            # チャンバー内部の円の半径を算出
        rr = round(1000 * math.sqrt(6 / math.pi) / 5)
        # 1.38^2*pi=6
        # 1000:(276)=5:1.38
        # マスクの作成
        img_mask = np.zeros([1400, 1400], np.uint8)
        # cv2.circle(img_mask,(cx,cy),r,(255,255,255),-1)
        cv2.circle(img_mask, (cx, cy), rr, (255, 255, 255), -1)
        # cv2.imwrite('black_img.jpg', img_mask)
        img_maskn = cv2.bitwise_not(img_mask)  # マスク画像の白黒を反転
        mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

        # 切り抜き
        masked_upstate = cv2.bitwise_and(dst, mask_rgb)
        masked_replace_white = cv2.addWeighted(masked_upstate, 1, cv2.cvtColor(img_maskn, cv2.COLOR_GRAY2RGB), 1, 0)
        dst_masked = cv2.cvtColor(masked_replace_white, cv2.COLOR_BGR2RGB)
        img_cut = dst_masked[cye - rr:cye + rr, cxe - rr:cxe + rr]

        plt.show(plt.imshow(img_cut))
        cv2.imwrite('masked_' + imagename, img_cut)

        # ヒストグラム
        n, bins, patches = plt.hist(img_cut[:, :, 1].ravel(), 256, range=(0, 255), fc='g')
        # n : 縦軸の値の数列
        # bins : 横軸の区切りとなる座標（各ビンの始点）
        # patches : patchの集合。
        maxId = signal.argrelmax(n, order=10)
        minmaxId = 256
        for x in maxId[0]:
            if minmaxId > x:
                minmaxId = x
            else:
                pass
        print(maxId)
        print(minmaxId)

        counterx = 0
        while counterx < minmaxId:
            countery = minmaxId - counterx
            if n[countery] <= 100:
                break
            else:
                counterx += 1
        print(countery)

        counterx2 = 0
        while counterx2 < 256 - minmaxId:
            countery2 = minmaxId + counterx2
            if n[countery2] <= 100:
                break
            else:
                counterx2 += 1
        print(countery2)

        # RGB分離
        img_blue_c1, img_green_c1, img_red_c1 = cv2.split(img_cut)
        ret, img_green_c1 = cv2.threshold(img_green_c1, countery2, 255, cv2.THRESH_BINARY)

        # モルフォロジー変換(オープニング，クロージング)
        kerne2 = np.ones((5, 5), np.uint8)
        opening_green = cv2.morphologyEx(img_green_c1, cv2.MORPH_OPEN, kerne2)
        # closing_green = cv2.morphologyEx(img_green_c2, cv2.MORPH_CLOSE, kerne2)
        closing_opening_green = cv2.morphologyEx(opening_green, cv2.MORPH_CLOSE, kerne2)
        # opening_closing_green = cv2.morphologyEx(closing_green, cv2.MORPH_OPEN, kerne2)

        img_mask2 = np.zeros([2 * rr, 2 * rr], np.uint8)
        cv2.circle(img_mask2, (rr, rr), rr, (255, 255, 255), -1)
        img_maskn2 = cv2.bitwise_not(img_mask2)  # マスク画像の白黒を反転
        closing_opening_green = cv2.add(img_maskn2, closing_opening_green)
        cv2.imwrite('thresholded_' + imagename, closing_opening_green)
        # cv2.imwrite('green_opening_closing_'+xi, opening_closing_green)

        check1 = cv2.bitwise_and(img_cut, cv2.cvtColor(closing_opening_green, cv2.COLOR_GRAY2RGB))
        check1 = cv2.add(cv2.cvtColor(img_maskn2, cv2.COLOR_GRAY2RGB), check1)
        cv2.imwrite('check1_' + imagename, check1)
        closing_opening_green2 = cv2.bitwise_not(closing_opening_green)  # マスク画像の白黒を反転
        check2 = cv2.bitwise_and(img_cut, cv2.cvtColor(closing_opening_green2, cv2.COLOR_GRAY2RGB))
        check2 = cv2.bitwise_and(check2, cv2.cvtColor(img_mask2, cv2.COLOR_GRAY2RGB))
        cv2.imwrite('check2_' + imagename, check2)

        area = 0
        for x in closing_opening_green:
            for y in x:
                if y == 0:
                    area += 1
                else:
                    pass
        print(area)
        areacm = area / 40000
        print(areacm)

        appendcsvlist = [imagename, area, areacm]
        csvlist.append(appendcsvlist)

    # ファイルオープン
    f = open('output_CR.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')

    # 出力
    writer.writerows(csvlist)

    # ファイルクローズ
    f.close()


function_perspective()
