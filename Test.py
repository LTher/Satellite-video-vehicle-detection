# -*- coding: utf-8 -*-

import cv2
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_uniform_float32
import time
import math

def init():
    a=0
# GPU function
@cuda.jit
def process_FirstFrame(rng_states,img,samples,dc_xoff,dc_yoff):
    rows, cols = img.shape
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    thread_id = cuda.grid(1)
    random=int(xoroshiro128p_uniform_float32(rng_states, thread_id)*9)
    col=tx+dc_xoff[random]
    if col<0:
        col=0
    if col>cols:
        col=cols-1
    row=ty+dc_yoff[random]
    if row<0:
        row=0
    if row>rows:
        row=rows-1
    #print(a)
    for i in range(0, 20):
        (samples[i])[col,row]=img[tx,ty]

@cuda.jit
def process_test(rng_states,img,samples,foregroundMatchCount,mask,dc_xoff,dc_yoff):
    thread_id = cuda.grid(1)
    rows, cols = img.shape
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    num_samples=20
    min_matches=3
    radius=30
    subsample_factor=16
    matches=0
    count=0
    if (tx<cols and ty<rows):
        while (matches < min_matches and count < num_samples):
            a = (samples[count])[tx, ty][0]
            b = img[tx, ty]
            c = b-a
            if(c>10000):
                c=18446744073709551615-c
            dist = abs(c)
            #print(dist)
            if (dist < radius):
                matches = matches + 1
            count=count+1
        if matches >= min_matches:
            #a=(samples[20])[tx, ty][0]
            #print(a)
            foregroundMatchCount[tx, ty] = 0
            mask[tx, ty] = 0
            random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * subsample_factor%subsample_factor)
            if random == 0:
                random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * num_samples%num_samples)
                (samples[random])[tx, ty] = img[tx, ty]
            random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * subsample_factor%subsample_factor)
            if random == 0:
                random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * 9)
                col = tx + dc_xoff[random]
                if col < 0:
                    col = 0
                if col > cols:
                    col = cols - 1
                row = ty + dc_yoff[random]
                if row < 0:
                    row = 0
                if row > rows:
                    row = rows - 1
                random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * num_samples%num_samples)
                (samples[random])[col, row] = img[tx, ty]
        else:
            foregroundMatchCount[tx, ty][0] = foregroundMatchCount[tx, ty][0] + 1
            mask[tx, ty] = 255
            if (foregroundMatchCount[tx, ty][0] > 50):
                random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * subsample_factor%subsample_factor)
                if random == 0:
                    random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * num_samples%num_samples)
                    (samples[random])[tx, ty] = img[tx, ty]

if __name__ == "__main__":
    ######################     视频载入       #############################
    cap = cv2.VideoCapture("D:\\V\\Test2.mp4")
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('E:\\Data_Set\\AODnet\\测试视频\\生成视频\\output11.avi', fourcc, 20, (1920, 1080))
    #####################      视频处理       #############################
    num = 0
    hc_xoff = [-1, 0, 1, -1, 1, -1, 0, 1, 0];  # x的邻居点，9宫格
    hc_yoff = [-1, 0, 1, -1, 1, -1, 0, 1, 0];  # y的邻居点
    dc_xoff = cuda.to_device(hc_xoff)
    dc_yoff = cuda.to_device(hc_yoff)
    blockspergrid=None
    threadsperblock=None
    rng_states=None
    d_samples=None
    d_foregroundMatchCount=None
    d_mask=None

    while cap.isOpened():
        # get a frame
        rval, frame = cap.read()
        # save a frame
        if rval == True:
            #  frame = cv2.flip(frame,0)
            # Start time
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows, cols = gray.shape
            #normalize
            gray_norm=np.zeros_like(gray)
            cv2.normalize(gray,gray_norm,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            #gamma
            fI=gray/255.0
            gamma=0.4
            gray_gamma=np.power(fI,gamma)
            cv2.normalize(gray_gamma, gray_gamma, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            dImg = cuda.to_device(gray_norm)
            start = time.time()
            #        rclasses, rscores, rbboxes=process_image(frame) #换成自己调用的函数

            # clean_image_tensor = process_image(data_hazy)  # 换成自己调用的函数
            if num==0:

                img_zeros = np.zeros([rows, cols, 1], np.uint8)
                img_zeros[:, :, 0] = np.zeros([rows, cols])
                d_mask = cuda.to_device(img_zeros)
                d_foregroundMatchCount = cuda.to_device(img_zeros)
                h_samples = []
                threadsperblock = (16, 16)
                blockspergrid_x = int(math.ceil(rows / threadsperblock[0]))
                blockspergrid_y = int(math.ceil(cols / threadsperblock[1]))
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                for i in range(0, 20):
                    h_samples.append(img_zeros)
                d_samples=cuda.to_device(h_samples)
                rng_states = create_xoroshiro128p_states(64 * blockspergrid_x * blockspergrid_y, seed=1)
                cuda.synchronize()
                process_FirstFrame[blockspergrid, threadsperblock](rng_states,dImg,d_samples,dc_xoff,dc_yoff)
                h_samples=d_samples.copy_to_host()
                '''for i in range(0, 20):
                    cv2.imshow("sample"+str(i),h_samples[i])
                    #cv2.imwrite("sample" + str(i) + ".jpg", h_samples[i], [int(cv2.IMWRITE_JPEG_QUALITY), 95])'''
                cuda.synchronize()
            else:
                cuda.synchronize()
                fore=dImg.copy_to_host()
                cv2.imshow("fore",fore)
                process_test[blockspergrid, threadsperblock](rng_states, dImg, d_samples, d_foregroundMatchCount, d_mask, dc_xoff, dc_yoff)
                cuda.synchronize()
                mask=d_mask.copy_to_host()
                '''kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
                mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)'''
                cv2.imshow("mask", mask)
            # End time
            end = time.time()
            # Time elapsed
            seconds = end - start + 0.0001
            print("Time taken : {0} seconds".format(seconds))
            # Calculate frames per second
            fps = 1 / seconds;
            print("Estimated frames per second : {0}".format(fps));
            # bboxes_draw_on_img(frame,rclasses,rscores,rbboxes)
            # print(rclasses)
            # out.write(clean_image)
            num = num + 1
            print(num)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        else:
            break
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

