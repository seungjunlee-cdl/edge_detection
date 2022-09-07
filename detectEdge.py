## code implementation of edge detection
import cv2
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convolution')
    parser.add_argument('-i',help='input image path')
    parser.add_argument('-k',help='convolution kernel. one of "roberts", "prewitt", "sobel", "LoG"')
    parser.add_argument('-o',help='output mask path')
    args = parser.parse_args()
    return args

def get_kernel(kernel):
    assert kernel in ['sobel','prewitt','roberts','LoG']

    kernelDict = {}
    kernelDict['roberts'] = [np.array([[1,0],[0,-1]]),np.array([[0,1],[-1,0]])]
    kernelDict['prewitt'] = [np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),np.array([[1,1,1],[0,0,0],[-1,-1,-1]])]
    kernelDict['sobel'] = [np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),np.array([[1,2,1],[0,0,0],[-1,-2,-1]])]
    kernelDict['LoG'] = [np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])]
    
    return kernelDict[kernel]

def detectEdge(inputImage,kernel):
    (H,W) = inputImage.shape
    edgeImage = np.zeros((H,W),dtype=int)

    for kernelObj in kernel:
        (kH,kW) = kernelObj.shape
        kernelMask = np.zeros((H,W),dtype=int)
        for convH in range(0,H-kH):
            for convW in range(0,W-kW):
                partImage = inputImage[convH:convH+kH,convW:convW+kW]
                partConv = np.abs(np.sum(partImage*kernelObj))
                kernelMask[convH,convW] = partConv

        edgeImage += kernelMask

    return edgeImage

def main():
    args = parse_args()
    inputImg = args.i
    kernel = args.k
    output = args.o

    image = cv2.imread(inputImg, cv2.IMREAD_UNCHANGED)
    kernelMatrix = get_kernel(kernel)
    mask = detectEdge(image,kernelMatrix)

    cv2.imwrite(output,mask)

if __name__ =='__main__':
    main()