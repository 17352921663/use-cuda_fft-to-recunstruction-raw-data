import numpy as np
import codecs
import struct
import time
import math
import matplotlib.pyplot as plt

from skimage import restoration
from scipy.fftpack import ifft2,fftshift, fft2
from skimage.filters import unsharp_mask , gaussian
from skimage.exposure import histogram , equalize_hist
from skimage.exposure import rescale_intensity
from skimage import img_as_uint

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import pydicom

def recon_gpu_2d(kdata):
    print(kdata.shape)
    imageData = np.empty(kdata.shape , np.complex128)

    for slice_idx in range(kdata.shape[2]):
        slice_data = kdata[:,:,slice_idx]
        data_gpu = gpuarray.to_gpu(slice_data)#ndarray to gpu data
        result_gpu = gpuarray.empty(slice_data.shape , np.complex128)
        plan_iverse = cu_fft.Plan(slice_data.shape , np.complex128 , np.complex128)

        cu_fft.ifft(data_gpu , result_gpu , plan_iverse , False)

        result = result_gpu.get()/slice_data.shape[0]/slice_data.shape[1]
        result = np.fft.fftshift(result ,1)
        imageData[:,:,slice_idx] = result
    return abs(imageData)

f = open('3644.raw','rb')
DataPosition=64512
f.seek(DataPosition,0)#0-head,1-this position,2-tail
NumSamples=f.read(4)
NumSamples = int.from_bytes(NumSamples,byteorder='little',signed=False)
#The highest byte is placed at the end of the byte array
f.seek(DataPosition+4,0)
NumViews=f.read(4)
NumViews = int.from_bytes(NumViews,byteorder='little',signed=False)
f.seek(DataPosition+8,0)
NumSecViews=f.read(4)
NumSecViews = int.from_bytes(NumSecViews,byteorder='little',signed=False)
f.seek(DataPosition+12,0)
NumSlices=f.read(4)
NumSlices = int.from_bytes(NumSlices,byteorder='little',signed=False)
f.seek(DataPosition+14,0)
DataTypes=f.read(4)
DataTypes = int.from_bytes(DataTypes,byteorder='little',signed=False)
f.seek(DataPosition+152,0)
NumEchos=f.read(4)
NumEchos = int.from_bytes(NumEchos,byteorder='little',signed=False)
f.seek(DataPosition+156,0)
NumExperiments=f.read(4)
NumExperiments = int.from_bytes(NumExperiments,byteorder='little',signed=False)
Count=2*NumSamples*NumViews*NumSecViews*NumSlices*NumEchos*NumExperiments
dataTmp=np.zeros(Count)
for i in range(Count):
    f.seek(65536 + 256 + 8+4*i, 0)
    Tmp = f.read(4)
    dataTmp[i] = int.from_bytes(Tmp, byteorder='little', signed=True)
DataTmpIm=dataTmp[0:Count:2]#Take a value every other line
DataTmpRe=dataTmp[1:Count:2]#
DataComplex=DataTmpIm+1j*DataTmpRe
DataMatrix=DataComplex.reshape((NumSamples,NumViews,NumSecViews,NumSlices,NumEchos,NumExperiments),order='F')
#Read vertically, write vertically, read / write one column first
## reconstruction
Data=DataMatrix.squeeze()
# mag=np.zeros([Data.shape[0],Data.shape[1],Data.shape[2]])*(1+1j)
# for slice in range(Data.shape[2]):
#     mag[:,:,slice]=np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Data[:,:,slice])),0))
# mag = mag.astype(np.int32)
result = recon_gpu_2d(Data)
plt.figure()
plt.axis('off')
plt.imshow(result[:,:,0], cmap='gray')
plt.show()