import cv2
import numpy as np

class MyBlender():
    '''
    Class that performs blending operations on images using pyramids.
    '''
    def __init__(self, depth=6):
        # 初始化混合器，设置金字塔的深度
        self.depth = depth
        
    def getGaussianPyramid(self, img):
        # 获取图像的高斯金字塔
        pyra = [img]  # 初始化金字塔列表，包含原始图像
        for i in range(self.depth - 1):
            down = cv2.pyrDown(pyra[i])  # 下采样图像
            pyra.append(down)  # 将下采样后的图像添加到金字塔列表中
        return pyra  # 返回高斯金字塔

    def getLaplacianPyramid(self, img):
        # 获取图像的拉普拉斯金字塔
        pyra = []  # 初始化金字塔列表
        for i in range(self.depth-1):
            nextImg = cv2.pyrDown(img)  # 下采样图像
            size = (img.shape[1], img.shape[0])  # 获取图像尺寸
            up = cv2.pyrUp(nextImg, dstsize=size)  # 上采样图像
            sub = img.astype(float) - up.astype(float)  # 计算拉普拉斯金字塔层
            pyra.append(sub)  # 将拉普拉斯层添加到金字塔列表中
            img = nextImg  # 更新图像为下采样后的图像
        pyra.append(img)  # 添加最后的下采样图像
        return pyra  # 返回拉普拉斯金字塔

    def getBlendingPyramid(self, lpa, lpb, gpm):
        # 根据掩码混合拉普拉斯金字塔的每一层
        pyra = []  # 初始化混合金字塔列表
        for i, mask in enumerate(gpm):
            maskNet = cv2.merge((mask, mask, mask))  # 合并掩码通道
            blended = lpa[i] * maskNet + lpb[i] * (1 - maskNet)  # 混合拉普拉斯层
            pyra.append(blended)  # 将混合后的层添加到金字塔列表中
        return pyra  # 返回混合金字塔

    def reconstruct(self, lp):
        # 从拉普拉斯金字塔重建图像
        img = lp[-1]  # 初始化图像为金字塔的最后一层
        for i in range(len(lp) - 2, -1, -1):
            laplacian = lp[i]  # 获取当前层的拉普拉斯图像
            size = laplacian.shape[:2][::-1]  # 获取图像尺寸
            img = cv2.pyrUp(img, dstsize=size).astype(float)  # 上采样图像
            img += laplacian.astype(float)  # 加上拉普拉斯图像
        return img  # 返回重建后的图像

    def getMask(self, img):
        # 获取图像的掩码
        mask = img[:, :, 0] != 0  # 检查红色通道是否为零
        mask = np.logical_and(img[:, :, 1] != 0, mask)  # 检查绿色通道是否为零
        mask = np.logical_and(img[:, :, 2] != 0, mask)  # 检查蓝色通道是否为零
        maskImg = np.zeros(img.shape[:2], dtype=float)  # 初始化掩码图像
        maskImg[mask] = 1.0  # 设置掩码图像的值
        return maskImg, mask  # 返回掩码图像和布尔掩码

    def blend(self, img1, img2, strategy='STRAIGHTCUT'):
        '''
        Blends the two images by getting the pyramids and blending appropriately.
        '''
        # 通过获取金字塔并适当混合来混合两张图像

        # 计算所需的拉普拉斯金字塔
        lp1 = self.getLaplacianPyramid(img1)  # 获取图像1的拉普拉斯金字塔
        lp2 = self.getLaplacianPyramid(img2)  # 获取图像2的拉普拉斯金字塔

        # 获取两张图像的掩码
        _, mask1truth = self.getMask(img1)  # 获取图像1的掩码
        _, mask2truth = self.getMask(img2)  # 获取图像2的掩码

        # 使用两张图像的重叠部分计算边界框
        yi, xi = np.where(mask1truth & mask2truth)  # 获取重叠区域的坐标
        overlap = mask1truth & mask2truth  # 计算重叠区域
        tempMask = np.zeros(img1.shape[:2])  # 初始化临时掩码
        yb, xb = np.where(overlap)  # 获取重叠区域的坐标
        minx = np.min(xb)  # 获取重叠区域的最小x坐标
        maxx = np.max(xb)  # 获取重叠区域的最大x坐标
        miny = np.min(yb)  # 获取重叠区域的最小y坐标
        maxy = np.max(yb)  # 获取重叠区域的最大y坐标
        h, w = tempMask.shape  # 获取图像高度和宽度

        finalMask = np.zeros(img1.shape[:2])  # 初始化最终掩码
        if strategy == 'STRAIGHTCUT':
            # 简单策略，如果只有从左到右的平移
            finalMask[:, :(minx + maxx) // 2] = 1.0  # 设置掩码的左半部分为1
        elif strategy == 'DIAGONAL':
            # 允许垂直移动的策略
            finalMask = cv2.fillConvexPoly(finalMask, np.array([
                [
                    [minx, miny], 
                    [maxx, maxy], 
                    [maxx, h], 
                    [0, h], 
                    [0, 0],
                    [minx, 0]
                ]
            ]), True, 50)  # 填充多边形掩码

        gpm = self.getGaussianPyramid(finalMask)  # 获取掩码的高斯金字塔
        
        blendPyra = self.getBlendingPyramid(lp1, lp2, gpm)  # 获取混合金字塔

        finalImg = self.reconstruct(blendPyra)  # 重建混合后的图像

        return finalImg, mask1truth, mask2truth  # 返回最终图像和掩码