# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/27 11:41
 @name: 
 @desc: frequencies difference
"""
import math
import numpy as np
import torch_dct as dct
import torch
import cv2
import scipy


# def delta(eeg, fold):
#     # time-series [t, d]
#     length = int(len(eeg) // fold)
#     base = eeg[0:length - 1, :]
#     a2 = eeg[2 * length:(length * 3) - 1, :]
#     assert np.shape(base)[0] == np.shape(a2)[0]
#     d = a2 - base
#     del eeg, base, a2
#     return d


def four_ave(eeg, fold):
    # [t, c]
    assert len(eeg) % fold == 0
    eeg = np.split(eeg, fold, axis=0)  # [fold length 96]
    eeg = np.average(eeg, axis=0)
    # assert np.shape(eeg) == (512, 96)
    return eeg


def delta_1(eeg):
    # [t c]
    assert len(eeg) % 4 == 0
    eeg = np.split(eeg, 4, axis=0)  # [fold t 96]
    base = eeg[0]
    d1 = eeg[1] - base
    d2 = eeg[2] - base
    d3 = eeg[3] - base
    re = (d1 + d2 + d3) / 3  # [t 96]
    # assert np.shape(re) == (1024, 96)
    return re


def delta_b(eeg):
    # [t c]
    assert len(eeg) % 4 == 0
    eeg = np.split(eeg, 4, axis=0)  # [fold t 96]
    base = eeg[3]
    d1 = eeg[1] - base
    d2 = eeg[2] - base
    d3 = eeg[0] - base
    re = (d1 + d2 + d3) / 3  # [t 96]
    # assert np.shape(re) == (1024, 96)
    return re


def stair_delta_ave(eeg):
    # [t c]
    assert len(eeg) % 4 == 0
    eeg = np.split(eeg, 4, axis=0)  # [fold t 96]
    d1 = eeg[1] - eeg[0]
    d2 = eeg[2] - eeg[1]
    d3 = eeg[3] - eeg[2]

    re = (d1 + d2 + d3) / 3  # [t 96]

    assert np.shape(re) == (512, 96)
    return re


def frame_stair_delta_ave(eeg):
    # [t c]
    assert len(eeg) % 4 == 0
    base = eeg[0::4, :]
    alph = eeg[1::4, :]
    bet = eeg[2::4, :]
    gam = eeg[3::4, :]

    d1 = alph - base
    d2 = bet - base
    d3 = gam - base

    re = (d1 + d2 + d3) / 3  # [t 96]

    assert np.shape(re) == (512, 96)
    return re


def frame_delta(eeg):
    # [t c]
    a = eeg[1:, :]
    b = eeg[:-1, :]
    assert len(a) == len(b)
    # assert np.shape(a) == (1024, 96)
    return a-b


def frame_delta_video(eeg):
    # [t c]
    a = eeg[1:, :, :]
    b = eeg[:-1, :, :]
    assert len(a) == len(b)
    # assert np.shape(a) == (1024, 96)
    return a-b


def noise_deactivate(eeg, threshold=0.1):
    # [t=512 c]
    ave = np.mean(eeg, axis=0)
    std = np.std(eeg, axis=0)

    re = eeg[:-1, :]
    re[np.abs(re) < threshold] = 0
    assert np.shape(re) == (512, 96)
    return re


def dct_1d(eeg, inverse=False):
    # [t d]
    eeg = torch.from_numpy(eeg).T.cuda()  # [d=96 t=512]
    sf = dct.dct(eeg)  # tensor [96 512]
    if inverse:
        fea_dim = sf.size()[-1]  # 512
        principle = fea_dim // 4  # 128
        fea_mask = torch.arange(1, fea_dim + 1)  # [512,]
        fea_mask = (fea_mask < principle).type(torch.uint8).to(sf.device)  # >高频  <低频
        # print(fea_mask)
        _sf = sf * fea_mask
        _s = dct.idct(_sf)  # [96 512]
        _s = _s.T  # [512 96]
        assert _s.size() == (512, 96)
        return _s.cpu().numpy()
    else:
        return sf.T.cpu().numpy()


def dct_1d_numpy(eeg, axis=-1, inverse=False):
    # [t d]
    # eeg = torch.from_numpy(eeg).T.cuda()  # [d=96 t=512]
    sf = scipy.fft.dct(eeg, axis=axis)
    # print(np.shape(sf), 'qqqqqqqqqqqqqqqqq')
    if inverse:
        fea_dim = sf.size()[-1]  # 512
        principle = fea_dim // 4  # 128
        fea_mask = np.arange(1, fea_dim + 1)  # [512,]
        fea_mask = (fea_mask < principle).type(np.uint8)  # >高频  <低频
        # print(fea_mask)
        _sf = sf * fea_mask
        _s = scipy.fft.idct(_sf)  # [96 512]
        assert _s.size() == (512, 96)
        return _s
    else:
        return sf


# def dct_2d(eeg, inverse=False):
#     # https://zhuanlan.zhihu.com/p/114626779
#     #  [t d]
#     freqs = cv2.dct(eeg)
#     if inverse:
#         # 裁剪DCT
#         for i in range(freqs.shape[0]):
#             for j in range(freqs.shape[1]):
#                 if i > 24 or j > 128:
#                     freqs[i, j] = 0  # 裁剪的实质为像素置0
#         return cv2.idct(freqs)
#     else:
#         # img_dct_log = 20 * np.log(abs(freqs))
#         return abs(freqs)

# 分块图 DCT 变换
def dct2d(img_f32, block=8):
    height, width = img_f32.shape[:2]
    block_y = height // block
    block_x = width // block
    height_ = block_y * block
    width_ = block_x * block
    img_f32_cut = img_f32[:height_, :width_]
    img_dct = np.zeros((height_, width_), dtype=np.float32)
    # new_img = img_dct.copy()
    for h in range(block_y):
        for w in range(block_x):
            # 对图像块进行dct变换
            img_block = img_f32_cut[block*h: block*(h+1), block*w: block*(w+1)]
            img_dct[block*h: block*(h+1), block*w: block*(w+1)] = cv2.dct(img_block)

            # 进行 idct 反变换
            # dct_block = img_dct[8*h: 8*(h+1), 8*w: 8*(w+1)]
            # img_block = cv2.idct(dct_block)
            # new_img[8*h: 8*(h+1), 8*w: 8*(w+1)] = img_block
    img_dct_log2 = np.log(abs(img_dct))
    return img_dct_log2


def approximated_dct(eeg):
    #  [t d]
    oo = ((eeg[::2, ::2] + eeg[1::2, ::2]) + (eeg[::2, 1::2] + eeg[1::2, 1::2])) * 2
    ol = ((eeg[::2, ::2] + eeg[1::2, ::2]) - (eeg[::2, 1::2] + eeg[1::2, 1::2])) * 2
    lo = ((eeg[::2, ::2] - eeg[1::2, ::2]) + (eeg[::2, 1::2] - eeg[1::2, 1::2])) * 2
    ll = ((eeg[::2, ::2] - eeg[1::2, ::2]) - (eeg[::2, 1::2] - eeg[1::2, 1::2])) * 2
    # [[oo, ol],
    #  [lo, ll]]
    assert np.shape(oo) == np.shape(ll) == (256, 48)
    a = np.concatenate([oo, ol], axis=1)
    b = np.concatenate([lo, ll], axis=1)
    re = np.concatenate([a, b], axis=0)
    assert np.shape(re) == (512, 96)
    return re


def down_sample(eeg, ratio):
    return eeg[::ratio, :]


def trial_average(eeg, axis=0):
    #  [1000, 127]
    ave = np.mean(eeg, axis=axis)
    std = np.std(eeg, axis=axis)
    return (eeg - ave) / std

def wrapping(eeg, n=3):
    eeg = np.where(eeg > 1, eeg + (1-n)/n, np.power(eeg, n)/n)
    return eeg

class Wrapping(torch.nn.Module):
    def __init__(self, b=0, n=3):
        super(Wrapping, self).__init__()
        self.b = b
        self.n = n

    def forward(self, x):
        return torch.where(x > 1, x + self.b + (1-self.n)/self.n, torch.pow((x+self.b), self.n)/self.n)


def DE_PSD(data, stft_para):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    # initialize the parameters
    STFTN = stft_para['stftn']
    fStart = stft_para['fStart']
    fEnd = stft_para['fEnd']
    fs = stft_para['fs']
    window = stft_para['window']

    WindowPoints = fs * window

    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    for i in range(0, len(stft_para['fStart'])):
        fStartNum[i] = int(fStart[i] / fs * STFTN)
        fEndNum[i] = int(fEnd[i] / fs * STFTN)

    # print(fStartNum[0],fEndNum[0])
    n = data.shape[0]
    m = data.shape[1]

    # print(m,n,l)
    # psd = np.zeros([n, len(fStart)])
    de = np.zeros([n, len(fStart)])
    # Hanning window
    Hlength = window * fs
    # Hwindow=hanning(Hlength)
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])

    WindowPoints = fs * window
    dataNow = data[0:n]
    for j in range(0, n):
        temp = dataNow[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = abs(FFTdata[0:int(STFTN / 2)])
        for p in range(0, len(fStart)):
            E = 0
            # E_log = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E = E + magFFTdata[p0] * magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            # psd[j][p] = E
            de[j][p] = math.log(100 * E, 2)
            # de(j,i,p)=log2((1+E)^4)

    return de


if __name__ == "__main__":
    x = np.random.random([2, 3, 2])
    print(trial_average(x, 1))
