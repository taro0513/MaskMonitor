import pandas as pd
import cv2
import time
import numpy as np
import sys
import numpy as np



def pHash(img,leng=32,wid=32):
    img = cv2.resize(img, (leng, wid))   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]            
    avreage = np.mean(dct_roi)
    phash_01 = (dct_roi>avreage)+0
    phash_list = phash_01.reshape(1,-1)[0].tolist()
    hash = ''.join([str(x) for x in phash_list])
    return hash

def dHash(img,leng=9,wid=8):
    img=cv2.resize(img,(leng, wid))
    image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash=[]
    for i in range(wid):
        for j in range(wid):
            if image[i,j]>image[i,j+1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def aHash(img,leng=8,wid=8):
    img=cv2.resize(img,(leng, wid))
    image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avreage = np.mean(image)                           
    hash = [] 
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            if image[i,j] >= avreage: 
                hash.append(1) 
            else: 
                hash.append(0) 
    return hash

def Hamming_distance(hash1,hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num 

def pp(image2, frame,m,n):
    start1 = time.time()
    d_dist = Hamming_distance(dHash(image2),dHash(frame))
    end1 = time.time()
    
    start2 = time.time()
    p_dist = Hamming_distance(pHash(image2),pHash(frame))
    end2 = time.time()
        
    start3 = time.time()
    a_dist = Hamming_distance(aHash(image2),aHash(frame))
    end3 = time.time()
    file = sys.stdout    
    sys.stdout = open('pic-1.txt', 'a+') 
    print(f"[M: {m}N: {n}]"+"="*100)
    print('a_dist is '+'%d' % a_dist + ', similarity is ' +'%f' % (1 - a_dist * 1.0 / 64) + ', time is ' +'%f' % (end3-start3))
    print('p_dist is '+'%d' % p_dist + ', similarity is ' +'%f' % (1 - p_dist * 1.0 / 64) + ', time is ' +'%f' % (end2-start2))
    print('d_dist is '+'%d' % d_dist + ', similarity is ' +'%f' % (1 - d_dist * 1.0 / 64) + ', time is ' +'%f' % (end1-start1))

    image1 = frame
    sys.stdout.close()   
    sys.stdout = file 
    return [(1 - a_dist * 1.0 / 64), (1 - p_dist * 1.0 / 64), (1 - d_dist * 1.0 / 64)]

similarity = []
import matplotlib.pyplot as plt

similarity = []

for i in range(1,5):    
    img = cv2.imread(f'pq/0{i}.jpg')
    print("i:",i)
    for k in range(1,5):
        # print("k:",k)
        img2 = cv2.imread(f'pq/0{k}.jpg')
        aa = pp(img, img2, i, k)
        print(aa)
        # aa ('1.00000', '1.00000')
        similarity.append(aa)
    # print(similarity)
similarity2 = []
for j in range(1,4):
    img = cv2.imread(f'pq/0{j}.jpg')
    print("j:",j,j+1)
    img2 = cv2.imread(f'pq/0{j+1}.jpg')
    aa = pp(img, img2, i, k)
    similarity2.append(aa)

similarity = np.array(similarity)
similarity2 = np.array(similarity2)
plt.figure(figsize=(8, 6))
print(len(similarity))
# plt.plot(np.arange(1, len(similarity) + 1), similarity[:,0], color = 'yellow', label = u'a hash')
# plt.plot(np.arange(1, len(similarity) + 1), similarity[:,1], color = 'red', label = u'p hash', linewidth=0.4)
# plt.plot(np.arange(1, len(similarity) + 1), similarity[:,2], color = 'blue', label = u'd hash', linewidth=0.4)
plt.plot(np.arange(1, len(similarity2) + 1), similarity2[:,1], color = 'red', label = u'p hash(DT)', linewidth=0.4)
plt.plot(np.arange(1, len(similarity2) + 1), similarity2[:,2], color = 'blue', label = u'd hash(DT)', linewidth=0.4)

plt.title('four p hash')
plt.legend()
plt.savefig(f'pq/fp.jpg')

plt.show()
plt.clf()