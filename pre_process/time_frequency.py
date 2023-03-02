# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/10/16 10:47
 @desc:
"""
import scipy.signal as scisgn
import pywt
import numpy as np
import matplotlib.pyplot as plt
import mne


def signal2spectrum_cwt(signal, time=None, width=None):
    # continues wavelet transform
    if time is None:
        time = np.linspace(0, 1, 500, endpoint=False)
    if width is None:
        widths = np.arange(12, 28)

    cwtmatr = scisgn.cwt(data=np.squeeze(signal), wavelet=scisgn.ricker, widths=widths)
    # plt.imshow(cwtmatr, extent=[1, 512, 12, 28], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    # plt.show()
    return cwtmatr


def signal2spectrum_pywt_cwt(signal, scales=np.arange(13, 28)):
    cwtmatr, freqs = pywt.cwt(data=signal, scales=scales, wavelet='mexh')
    # image = Image.fromarray(np.uint8(cwtmatr))
    # plt.imshow(image)
    # plt.imshow(cwtmatr, extent=[1, 512, 1, 13], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    # plt.show()
    return cwtmatr


# def cwt_on_sample_thread(sample):
#     # [c=96, t=512]
#
#
#     return cwtmatrs


def cwt_on_sample(sample):
    # [c=96, t=512]
    Ws = mne.time_frequency.tfr.morlet(sfreq=1024, freqs=np.arange(12, 28, 1), n_cycles=5)  # 1/lowest*n = time
    tfr = mne.time_frequency.tfr.cwt(X=sample, Ws=Ws, use_fft=True)

    cwtmatrs = np.real(tfr)
    print(np.shape(cwtmatrs))
    plt.imshow(np.real(cwtmatrs[0]), extent=[1, 1024, 1, 97], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatrs[0]).max(), vmin=-abs(cwtmatrs[0]).max())
    plt.show()
    return cwtmatrs


# def pywt_dwt(signal):
#     wavename = 'db5'
#     cA, cD = pywt.dwt(np.squeeze(signal), wavename)
#
#     ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
#     yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
#
#     x = range(len(np.squeeze(mat['AF3'])))
#     plt.figure(figsize=(12, 9))
#     plt.subplot(311)
#     plt.plot(x, np.squeeze(mat['AF3']))
#     plt.title('original signal')
#     plt.subplot(312)
#     plt.plot(x, ya)
#     plt.title('approximated component')
#     plt.subplot(313)
#     plt.plot(x, yd)
#     plt.title('detailed component')
#     plt.tight_layout()
#     plt.show()
#     return ya, yd


# def plot_wavelet(time, signal, scales,
#                  waveletname='mexh',
#                  cmap=plt.cm.seismic,
#                  title='Wavelet Transform (Power Spectrum) of signal',
#                  ylabel='Period',
#                  xlabel='Time'):
#     dt = time[1] - time[0]
#     print(pywt.wavelist())
#     [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
#
#     power = (abs(coefficients)) ** 2
#     period = 1. / frequencies
#     levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
#     contourlevels = np.log2(levels)
#
#     fig, ax = plt.subplots(figsize=(15, 10))
#     im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)
#
#     ax.set_title(title, fontsize=20)
#     ax.set_ylabel(ylabel, fontsize=18)
#     ax.set_xlabel(xlabel, fontsize=18)
#
#     yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
#     ax.set_yticks(np.log2(yticks))
#     ax.set_yticklabels(yticks)
#     ax.invert_yaxis()
#     ylim = ax.get_ylim()
#     ax.set_ylim(ylim[0], -1)
#
#     cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
#     fig.colorbar(im, cax=cbar_ax, orientation="vertical")
#     plt.show()


# from pre_process.time_frequency import  *
# N = np.shape(signal)[0]
# t0 = 0
# dt = 0.25
# time = np.arange(0, N)
# scales = np.arange(0, 96)
# plot_signal_plus_average(time, signal)
# # plot_fft_plus_power(time, signal)
# plot_wavelet(time, signal, scales)

# def get_ave_values(xvalues, yvalues, n = 5):
#     signal_length = len(xvalues)
#     if signal_length % n == 0:
#         padding_length = 0
#     else:
#         padding_length = n - signal_length//n % n
#     xarr = np.array(xvalues)
#     yarr = np.array(yvalues)
#     xarr.resize(signal_length//n, n)
#     yarr.resize(signal_length//n, n)
#     xarr_reshaped = xarr.reshape((-1,n))
#     yarr_reshaped = yarr.reshape((-1,n))
#     x_ave = xarr_reshaped[:,0]
#     y_ave = np.nanmean(yarr_reshaped, axis=1)
#     return x_ave, y_ave


# def plot_signal_plus_average(time, signal, average_over=5):
#     fig, ax = plt.subplots(figsize=(15, 3))
#     time_ave, signal_ave = get_ave_values(time, signal, average_over)
#     ax.plot(time, signal, label='signal')
#     ax.plot(time_ave, signal_ave, label='time average (n={})'.format(5))
#     ax.set_xlim([time[0], time[-1]])
#     ax.set_ylabel('Signal Amplitude', fontsize=18)
#     ax.set_title('Signal + Time Average', fontsize=18)
#     ax.set_xlabel('Time', fontsize=18)
#     ax.legend()
#     plt.show()


# def get_fft_values(y_values, T, N, f_s):
#     f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
#     fft_values_ = fft(y_values)
#     fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
#     return f_values, fft_values
#
#
# def plot_fft_plus_power(time, signal):
#     dt = time[1] - time[0]
#     N = len(signal)
#     fs = 1 / dt
#
#     fig, ax = plt.subplots(figsize=(15, 3))
#     variance = np.std(signal) ** 2
#     f_values, fft_values = get_fft_values(signal, dt, N, fs)
#     fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
#     ax.plot(f_values, fft_values, 'r-', label='Fourier Transform')
#     ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
#     ax.set_xlabel('Frequency [Hz / year]', fontsize=18)
#     ax.set_ylabel('Amplitude', fontsize=18)
#     ax.legend()
#     plt.show()
