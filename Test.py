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
def process_background(cur_img,fore_img,Iback,Ibackmask,count):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    c=cur_img[tx,ty]-fore_img[tx,ty]
    if (c > 10000):
        c = 18446744073709551615 - c
    Idiff=abs(c)
    if Idiff==0 and Iback[tx,ty][0]==0:
        Ibackmask[tx,ty]=1
        Iback[tx,ty]=cur_img[tx,ty]
    if Iback[tx,ty][0]==0:
        count[0]=count[0]+1

@cuda.jit
def porcess_background2(img,Iback,count):
    rows, cols = img.shape
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if Iback[tx,ty][0]==0:
        value=0
        sum = 9
        for i in range(-1,2):
            for j in range(-1,2):
                col=tx+i
                if col < 0:
                    col = 0
                if col > cols:
                    col = cols - 1
                row=ty+j
                if row < 0:
                    row = 0
                if row > rows:
                    row = rows - 1
                if Iback[col,row][0]==0:
                    sum=sum-1
                else:
                    value=value+Iback[col,row][0]
        if sum!=0:
            value=int(value/sum)
        if value==0:
            count[1]=count[1]+1
        Iback[tx,ty][0]=value

@cuda.jit
def process_FirstFrame(rng_states,rcimg,img,samples,dc_xoff,dc_yoff):
    rows, cols = rcimg.shape
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    thread_id = cuda.grid(1)
    random=int(xoroshiro128p_uniform_float32(rng_states, thread_id)*9)
    col=tx+dc_xoff[random]
    if col<0:
        col=0
    if col>cols:
        col=cols-1
    random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * 9)
    row=ty+dc_yoff[random]
    if row<0:
        row=0
    if row>rows:
        row=rows-1
    #print(a)
    for i in range(0, 20):
        (samples[i])[col,row]=img[tx,ty][0]

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

@cuda.jit
def process_test_for_light(rng_states,img,samples,foregroundMatchCount,mask,dc_xoff,dc_yoff,frame_count):
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
            #foregroundMatchCount[tx, ty] = 0
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
            d_fore_count=frame_count[0]-foregroundMatchCount[tx, ty][0]
            thread=frame_count[0]*0.7
            #print(d_fore_count)
            if frame_count[0]>5 and d_fore_count < thread:
                mask[tx, ty] = 0
                random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * subsample_factor%subsample_factor)
                if random == 0:
                    random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * num_samples%num_samples)
                    (samples[random])[tx, ty] = img[tx, ty]

@cuda.jit
def process_test_for_noise(rng_states,img,samples,foregroundMatchCount,mask,dc_xoff,dc_yoff,frame_count,continuous_img,result):
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
    dyn=255
    if (tx<cols and ty<rows):
        while (matches < min_matches and count < num_samples):
            a = (samples[count])[tx, ty][0]
            b = img[tx, ty]
            c = b-a
            if(c>10000):
                c=18446744073709551615-c
            dist = abs(c)
            if dist<dyn:
                dyn=dist
            #print(dist)
            if (dist < radius):
                matches = matches + 1
            count=count+1
        frame_count[1]=frame_count[1]+dyn
        if matches >= min_matches:
            #a=(samples[20])[tx, ty][0]
            #print(a)
            #foregroundMatchCount[tx, ty] = 0
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
            d_fore_count=frame_count[0]-foregroundMatchCount[tx, ty][0]
            thread=frame_count[0]*0.7
            #print(d_fore_count)
            if frame_count[0]>5 and d_fore_count < thread:
                mask[tx, ty] = 0
                random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * subsample_factor%subsample_factor)
                if random == 0:
                    random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * num_samples%num_samples)
                    (samples[random])[tx, ty] = img[tx, ty]
            else:
                mask[tx, ty] = 255
                result[tx, ty][0] = 0
                result[tx, ty][1] = 0
                result[tx, ty][2] = 255
            '''if continuous_img[tx, ty][0]==0:
                mask[tx, ty] = 0
                random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * subsample_factor % subsample_factor)
                if random == 0:
                    random = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * num_samples % num_samples)
                    (samples[random])[tx, ty] = img[tx, ty]'''

@cuda.jit
def process_result(src_img,mask,result):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if mask[tx,ty][0]==255:
        result[tx,ty][0]=0
        result[tx,ty][1]=0
        result[tx,ty][2]=255



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

    count=np.zeros(2)
    count[0]=3840*2160
    count[1]=10
    d_count=cuda.to_device(count)
    thread1=count[0]/500

    frame_count = np.zeros(3)
    frame_count[0] = 0
    frame_count[2]=7
    d_frame_count = cuda.to_device(frame_count)

    ifdone=0

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
            fI = float(1.5)*fI
            fI[fI>1]=1
            gamma=2.4
            gray_gamma=np.power(fI,gamma)
            gray_gamma=gray_gamma*255.0
            #cv2.imwrite("D:\\PyProjects\\ViBe_CUDA\\gray\\gray_gmma.jpg",gray_gamma)
            cv2.normalize(gray_gamma, gray_gamma, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            gray_gamma = gray_gamma.astype(np.uint8)
            dImg = cuda.to_device(gray)

            src_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            #d_result = cuda.to_device(src_img)
            d_result = cuda.to_device(frame)
            start = time.time()
            #        rclasses, rscores, rbboxes=process_image(frame) #换成自己调用的函数

            # clean_image_tensor = process_image(data_hazy)  # 换成自己调用的函数
            '''if num==0:
                ave_img=gray.astype(np.float32)
                gray_final = dImg.copy_to_host()
                cv2.imwrite("D:\\PyProjects\\ViBe_CUDA\\gray\\gray_final.jpg", gray_final)
            elif num<19:
                temp_img=gray.astype(np.float32)
                ave_img=ave_img+temp_img
            elif num==19:
                temp_img = gray.astype(np.float32)
                ave_img = ave_img + temp_img
                ave_img=ave_img/20
                ave_img=ave_img.astype(np.uint8)

                cv2.imshow("backimg",ave_img)
                dImg=cuda.to_device(ave_img)
            
                origin first frame
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
                samples
                for i in range(0, 20):
                    cv2.imshow("sample"+str(i),h_samples[i])
                    #cv2.imwrite("sample" + str(i) + ".jpg", h_samples[i], [int(cv2.IMWRITE_JPEG_QUALITY), 95])'''
                #cuda.synchronize()
            if num==0:
                fore_frame=dImg.copy_to_host()
                img_zeros = np.zeros([rows, cols, 1], np.uint8)
                img_zeros[:, :, 0] = np.zeros([rows, cols])
                d_Iback = cuda.to_device(img_zeros)
                d_Ibackmask = cuda.to_device(img_zeros)
                threadsperblock = (16, 16)
                blockspergrid_x = int(math.ceil(rows / threadsperblock[0]))
                blockspergrid_y = int(math.ceil(cols / threadsperblock[1]))
                blockspergrid = (blockspergrid_x, blockspergrid_y)

                continuous_img=np.zeros([rows, cols, 1], np.uint8)
                continuous_img[:, :, 0] = np.ones([rows, cols])*255
                d_continuous_img=cuda.to_device(continuous_img)
                cv2.imshow("continous",continuous_img)

                fore = dImg.copy_to_host()
                # cv2.imwrite("fore.jpg",fore)
                cv2.imshow("fore", fore)

            elif num!=0 and count[0]>thread1:
                count[0]=0
                d_count=cuda.to_device(count)
                cur_frame=dImg.copy_to_host()
                d_foreframe=cuda.to_device(fore_frame)
                d_curframe=cuda.to_device(cur_frame)
                cuda.synchronize()
                process_background[blockspergrid, threadsperblock](d_curframe,d_foreframe,d_Iback,d_Ibackmask,d_count)
                cuda.synchronize()
                count=d_count.copy_to_host()
                print("count: ")
                print(count)
                print("processing background111111111")
                d_foreframe=cuda.to_device(cur_frame)

                Iback=d_Iback.copy_to_host()
                cv2.imshow("Iback",Iback)

                fore = dImg.copy_to_host()
                # cv2.imwrite("fore.jpg",fore)
                cv2.imshow("fore", fore)

            elif count[0]<=thread1 and count[1]!=0:
                count[1]=0
                d_count=cuda.to_device(count)
                cuda.synchronize()
                porcess_background2[blockspergrid, threadsperblock](dImg,d_Iback,d_count)
                cuda.synchronize()
                count=d_count.copy_to_host()

                Iback = d_Iback.copy_to_host()
                cv2.imshow("Iback", Iback)

                print("count: ")
                print(count)
                print("processing background2222222222")
                cuda.synchronize()
                if count[1]==0:
                    img_zeros = np.zeros([rows, cols, 1], np.uint8)
                    img_zeros[:, :, 0] = np.zeros([rows, cols])
                    d_mask = cuda.to_device(img_zeros)
                    d_foregroundMatchCount = cuda.to_device(img_zeros)
                    h_samples = []

                    for i in range(0, 20):
                        h_samples.append(Iback)
                    d_samples = cuda.to_device(h_samples)
                    rng_states = create_xoroshiro128p_states(64 * blockspergrid_x * blockspergrid_y, seed=1)
                    cuda.synchronize()
                    #d_Iback
                    process_FirstFrame[blockspergrid, threadsperblock](rng_states, dImg, d_Iback, d_samples, dc_xoff, dc_yoff)
                    #src
                    #process_FirstFrame[blockspergrid, threadsperblock](rng_states, dImg, dImg, d_samples, dc_xoff, dc_yoff)
                    h_samples = d_samples.copy_to_host()
                    cuda.synchronize()



                fore = dImg.copy_to_host()
                # cv2.imwrite("fore.jpg",fore)
                cv2.imshow("fore", fore)


            else:
                cuda.synchronize()
                fore=dImg.copy_to_host()
                #cv2.imwrite("fore.jpg",fore)
                cv2.imshow("fore",gray)

                #process_test[blockspergrid, threadsperblock](rng_states, dImg, d_samples, d_foregroundMatchCount, d_mask, dc_xoff, dc_yoff)
                frame_count=d_frame_count.copy_to_host()
                frame_count[0]=frame_count[0]+1
                d_frame_count=cuda.to_device(frame_count)
                #process_test_for_light[blockspergrid, threadsperblock](rng_states, dImg, d_samples, d_foregroundMatchCount, d_mask, dc_xoff, dc_yoff,d_frame_count)
                process_test_for_noise[blockspergrid, threadsperblock](rng_states, dImg, d_samples, d_foregroundMatchCount, d_mask, dc_xoff, dc_yoff, d_frame_count,d_continuous_img,d_result)
                cuda.synchronize()
                mask=d_mask.copy_to_host()
                '''kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
                mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)'''

                cv2.imshow("mask", mask)

                #cv2.imwrite("D:\\PyProjects\\ViBe_CUDA\\fore\\fore" + str(num) + ".jpg", fore)
                #cv2.imwrite("D:\\PyProjects\\ViBe_CUDA\\mask\\mask" + str(num) + ".jpg", mask)
                '''result = d_result.copy_to_host()
                cv2.imshow("result", result)

                cv2.imwrite("D:\\PyProjects\\ViBe_CUDA\\mask\\mask" + str(num) + ".bmp", mask)
                cv2.imwrite("D:\\PyProjects\\ViBe_CUDA\\mask\\result" + str(num) + ".bmp", result)
                cv2.imwrite("D:\\PyProjects\\ViBe_CUDA\\mask\\frame" + str(num) + ".bmp", frame)'''

                '''src_img=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
                cuda.synchronize()
                d_result=cuda.to_device(src_img)
                d_src=cuda.to_device(src_img)
                process_result[blockspergrid, threadsperblock](d_src,d_mask,d_result)
                cuda.synchronize()
                result=d_result.copy_to_host()
                cv2.imshow("result",result)'''
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

