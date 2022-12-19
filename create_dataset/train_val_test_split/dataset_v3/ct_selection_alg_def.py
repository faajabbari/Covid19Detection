#This code presents the CT scans selection algortihm from https://github.com/mr7495/COVID-CT-Code
import os
import numpy as np
import cv2
import shutil


def find_inds(ch,st): #function to find the indexes of a character in a string
    indexes=[]
    index0=0
    num=0
    while(True):
        if ch in st[num:]:
            index0=st[num:].index(ch)
            indexes.append(index0+num)
            num=num+index0+1
        else:
            break
    return(indexes)

#
# original_path=  '/home/fatemeh/Data_sets/our/our_data/chest_normal/converted_nii2png/1/4_lung_15__u91s'  #'COVID-CTset' #Path to the original dataset
# save_path = os.path.join('/home/fatemeh/Desktop','selected_TIF')
def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None


def ct_selection_alg(original_path, save_path):
    #os.makedirs(save_path, exist_ok=True)
    adds={}
    import pudb; pu.db
    for r,d,f in os.walk(original_path): #Take the addresses of the TIFF files for each patient in the dataset
        for file in f:
            if '.png' in file:
                full_add=os.path.join(r,file)
                indexes=find_inds('/',full_add)
                index=indexes[-1]
                if full_add[:index+1] not in adds:
                    adds[full_add[:index+1]]=[]
                adds[full_add[:index+1]].append(full_add[index+1:])

    selected={}
    for key in adds:
        try:

            zero=[]
            names=[]
            iss = []
            for value in adds[key]:

                names.append(value)
                address=key+value
                pixel=cv2.imread(address, 0) #read the TIFF file
                #if value == 's3___155___w-600___6039387432__non_covid__non_covid.png':
                 #   import pudb;pu.db
                #pixel = cv2.imread('/mnt/data/jj.png', 0)
                sp=pixel[240:340,120:370] #Crop the region
                
                counted_zero=0
                import pudb; pu.db                
                reshaped = np.reshape(sp,(sp.shape[0]*sp.shape[1],1))
                counted_zero = np.sum(reshaped < 60)
                zero.append(counted_zero)
            min_zero=min(zero)
            max_zero=max(zero)
            # print(min_zero)
            # print(max_zero)
            threshold=(max_zero-min_zero)/1.1 #Set the threshold
            indices=np.where(np.array(zero)>threshold) #Find the images that have more dark pixels in the region than the calculated threshold
            selected_names=np.array(names)[indices]
            selected[key]=list(selected_names) #Add the selected images of each patient
        except Exception as e:
            print(e)

    #Copy the selected images to the new folder
    paths = []
    for key in selected:
       # indexes=find_inds('/',key)
       # for i in range(4):
       #     globals()['index{}'.format(i+1)]=indexes[i]
       # try:
       #     os.mkdir('selected_TIF/{}'.format(key[index1+1:index2]))
       # except:
       #     pass
       # try:
       #     os.mkdir('selected_TIF/{}/{}'.format(key[index1+1:index2],key[index2+1:index3]))
       # except:
       #     pass
       # try:
       #     os.mkdir('selected_TIF/{}/{}/{}'.format(key[index1+1:index2],key[index2+1:index3],key[index3+1:index4]))
       # except:
       #     pass
        for value in selected[key]:
            address=key+value
            #print('kjghkjh',address)
            paths.append(address)
            # new_address='selected_TIF'+address[3:]
            #if key.split('/')[-2] == '1.3.12.2.1107.5.1.4.78983.30000020032114281185900000653':
             #   import pudb;pu.db
            #dest = '/mnt/data/ct3'
            #shutil.copy(address,dest)
    print('one item end')
    return paths


# ct_selection_alg(original_path, save_path)
