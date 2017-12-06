import numpy as np
import scipy.ndimage.interpolation as interp

def gen_synthetic(dataset):

    synthesized = []

    for d in dataset:
        num_images = np.random.poisson(1)
        for i in range(num_images):
            choice = np.random.randint(2)
            if choice == 0:
                ang = np.random.uniform(0., 360.)
                img = rotate(d[0], ang)
            elif choice == 1:
                fac = np.random.uniform(0.5, 1)
                img = scale(d[0], fac)
            synthesized.append((img, d[1]))

    return synthesized


def rotate(data, ang):
   datarot = interp.rotate(data, ang, axes=(1, 2), mode='nearest')
   margin = (datarot.shape[1] - data.shape[1]) // 2
   return datarot[:, margin:margin + data.shape[1], margin:margin + data.shape[2]]


def scale(data, fac):
    datascale = interp.zoom(data, fac, mode='nearest')
    if fac >= 1:
        margin = (datascale.shape[1] - data.shape[1])//2
        marginx = (datascale.shape[0] - data.shape[0])//2
        return datascale[marginx:marginx+data.shape[0],margin:margin+data.shape[1], margin:margin+data.shape[2]]
    else:
        margin = (data.shape[1]-datascale.shape[1])//2
        marginx = (data.shape[0]-datascale.shape[0])//2
        filldata = np.zeros(data.shape)
        filldata[marginx:marginx+datascale.shape[0],margin:margin+datascale.shape[1],margin:margin+datascale.shape[2]] = datascale
        filldata[marginx:marginx+datascale.shape[0],:margin,margin:margin+datascale.shape[2]] = datascale[:,0,:][:,np.newaxis,:]
        filldata[marginx:marginx+datascale.shape[0],margin+datascale.shape[1]:,margin:margin+datascale.shape[2]] = datascale[:,-1,:][:,np.newaxis,:]
        filldata[:,:,:margin] = filldata[:,:,margin][:,:,np.newaxis]
        filldata[:,:,margin+datascale.shape[2]:] = filldata[:,:,margin+datascale.shape[2]-1][:,:,np.newaxis]
        filldata[:marginx,:,:] = filldata[marginx,:,:][np.newaxis,:,:]
        filldata[marginx+datascale.shape[0]:,:,:] = filldata[marginx+datascale.shape[0]-1,:,:][np.newaxis,:,:]
        return filldata