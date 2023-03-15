# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/16 10:47
 @desc:
"""
import pywt
import numpy as np
import einops
import matplotlib.pyplot as plt
import scipy
import torch


def stft_torch(signal, n_fft=78, hop_length=20):
    signal = torch.tensor(signal, dtype=torch.float)
    y = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft),
                   center=True, return_complex=True)
    y_real = torch.view_as_real(y)[:, :, 0]
    return y_real.numpy()  # [f, t]


def stft_scipy(signal, nperseg=64):
    f, t, Zxx = scipy.signal.stft(signal, fs=1000, nperseg=nperseg, return_onesided=True)
    return f, t, Zxx  # [f t]


def spectrogram_scipy(signal, fs=1024, nperseg=64, overlap=0.5):
    # [t c]
    specs = []
    for s in signal.T:
        _, _, zxx = scipy.signal.spectrogram(s, fs=fs, nperseg=nperseg, noverlap=int(nperseg * overlap),
                                             scaling='spectrum', return_onesided=True)
        assert np.shape(zxx) == (33, 63)  # Purdue
        specs.append(zxx)
    return np.array(specs)  # [c f t]


def three_bands(signal):
    specs = spectrogram_scipy(signal=signal)  # [c f t]
    theta = np.sum(np.square(abs(specs[:, 0:8, :])), axis=1)  # [c 7 t] -> [c t] (96, 63)
    alpha = np.sum(np.square(abs(specs[:, 8:14, :])), axis=1)
    beta = np.sum(np.square(abs(specs[:, 14:, :])), axis=1)
    re = np.concatenate((theta, alpha, beta), axis=0)  # [3*c t]
    assert np.shape(re) == (3*96, 63)
    return re.T  # [63, 3*96]


def cwt_pywt(signal, wavelet='morl'):
    # fc = pywt.central_frequency(wavelet)  # 计算小波函数的中心频率
    # cparam = 2 * fc * totalscal  # 常数c
    # # 可以用 *f = scale2frequency(wavelet, scale)/sampling_period 来确定物理频率大小。f的单位是赫兹，采样周期的单位为秒。
    # scales = cparam / np.arange(totalscal+1, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）

    dt = 0.001  # 1000Hz
    fs = 1 / dt
    interested = np.array(range(40, 0, -1))
    frequencies = interested / fs  # normalize
    scale = pywt.frequency2scale(wavelet=wavelet, freq=frequencies)
    cwtmatr, _ = pywt.cwt(data=signal, scales=scale, wavelet=wavelet, sampling_period=0.001)
    return np.real(cwtmatr)  # [f, t]


# def cwt_on_sample(sample):
#     # [c=96, t=512]
#     Ws = mne.time_frequency.tfr.morlet(sfreq=1024, freqs=np.arange(12, 28, 1), n_cycles=5)  # 1/lowest*n = time
#     tfr = mne.time_frequency.tfr.cwt(X=sample, Ws=Ws, use_fft=True)
#
#     cwtmatrs = np.real(tfr)
#     print(np.shape(cwtmatrs))
#     plt.imshow(np.real(cwtmatrs[0]), extent=[1, 1024, 1, 97], cmap='PRGn', aspect='auto',
#                vmax=abs(cwtmatrs[0]).max(), vmin=-abs(cwtmatrs[0]).max())
#     plt.show()
#     return cwtmatrs


if __name__ == "__main__":
    import pickle
    import numpy as np
    import PIL.Image as Image

    print(pywt.wavelist(family=None, kind='continuous'))
    filepath = 'G:/Datasets/SZUEEG/EEG/pkl_ave/run_1_test_hzy_2_6501_38.pkl'
    # filepath = 'G:/Datasets/SZUEEG/EEG/pkl_ave/run_1_test_hzy_66_195501_38.pkl'
    # filepath = 'G:/Datasets/SZUEEG/EEG/pkl_ave/run_1_test_hzy_4_12501_18.pkl'
    # filepath = 'G:/Datasets/SZUEEG/EEG/pkl_ave/run_1_test_hzy_88_261501_18.pkl'
    # t = np.arange(0, 2, 1.0 / 1000)
    with open(filepath, 'rb') as f:
        x = pickle.load(f)  # SZU: [t=2000, channels=127], Purdue: [512, 96]
        print('xxxx', np.shape(x))
        y = int(pickle.load(f))

        # cwtmatr = signal2spectrum_pywt_cwt(x[:, 0])
        # plt.contourf(t, np.arange(40, 0, -1), abs(cwtmatr))

        # cwtmatr = stft_torch(x[:, 0])
        f, t, cwtmatr = spectrogram_scipy(x[:, 0])
        # from cwt_torch import CWT
        # pycwt = CWT(dj=0.0625, dt=1/1000, fmin=1, fmax=20, output_format=False)
        # x = torch.tensor(x, dtype=torch.float32).permute(1, 0).unsqueeze(0)
        # cwtmatr = pycwt(x)
        # cwtmatr = cwtmatr[:, :, :, :, 1]
        # print(cwtmatr.size())
        # cwtmatr = cwtmatr.numpy()[0, 0, :, :]
        plt.contourf(t, f, abs(cwtmatr))

        # image = Image.fromarray(np.uint8(cwtmatr))
        # plt.imshow(image)

        print(np.shape(cwtmatr))
        plt.show()

