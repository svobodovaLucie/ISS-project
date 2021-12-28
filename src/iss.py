#####################################
#             VUT FIT               #
#          ISS  2021/2022           #
#         Lucie Svobodová           #
#             xsvobo1x              #
#####################################

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

def plot_graph(time, title, filename, xlabel, ylabel, data):
  plt.figure(figsize=(7,4))
  plt.title(title)
  plt.plot(time, data)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(f'{filename}.pdf')

def load_file():
  fs, data = wavfile.read('../audio/xsvobo1x.wav')
  # normalise
  data = data / 2**15

  # data info
  print('Sample rate: ', fs)
  print('Length: ', data.shape[0], ' samples')
  print('Length: ', data.shape[0]/fs, 'sec')
  print('Min: ', data.min())
  print('Max: ', data.max())

  # plot the graph
  time = np.linspace(0, len(data)/fs, num=len(data))
  plot_graph(time, 'Vstupní signál', 'task1', 'Čas [s]', 'Amplituda', data)

  return data, fs

def task_2(data, fs):
  # centering the signal using mean value
  data = data - np.mean(data)

  # normalise the data
  data_abs = np.absolute(data)
  data = data/max(data_abs)

  # create frames
  frame_len = 1024
  overlap = 512
  frames = [[0] * frame_len for i in range((len(data)//frame_len) * 2 - 1)]
  j = 0
  for i in range(0, len(data) - overlap, overlap):
    frames[j] = data[i : i+frame_len]
    j += 1

  # saving frames as matrix and transpose
  frames = np.array(frames)
  frames = frames.transpose()

  # plotting one frame
  frame_num = 24
  nice_frame = frames[0:1024, frame_num]
  
  time = np.linspace(0, len(nice_frame)/fs, num=len(nice_frame))
  plot_graph(time, 'Rámec č. 24', 'task2', 'Čas [s]', 'Amplituda', nice_frame)

  return nice_frame

def DFT(data, fs):
  n = np.arange(len(data))
  k = n.reshape((len(data), 1))
  M = np.exp(-2j * np.pi * k * n / len(data))
  res = np.dot(M, data)
  # plot my DFT
  time = np.linspace(0,fs/2, num=len(data)//2)
  plot_graph(time, 'DFT (rámec č. 24)', 'task3_1', 'Frekvence [Hz]', 'Amplituda', np.abs(res[0:len(data)//2]))

  # plot Python DFT (FFT)
  data = np.fft.fft(data)

  time = np.linspace(0,fs/2, num=len(data)//2)
  plot_graph(time, 'DFT (rámec č. 24) - numpy.fft.fft()', 'task3_2', 'Frekvence [Hz]', 'Amplituda', np.abs(data[0:len(data)//2]))

def plot_spectrogram(data, fs, filename, title):
  plt.figure(figsize=(7,4))
  plt.tight_layout()
  plt.specgram(data, Fs=fs, NFFT=1024, noverlap=512, mode='psd', scale='dB')
  plt.title(title)
  plt.gca().set_xlabel('Čas [s]')
  plt.gca().set_ylabel('Frekvence [Hz]')
  cbar = plt.colorbar()
  cbar.set_label('Spektrální hustota výkonu [dB]', rotation=270, labelpad=15)
  plt.savefig(f'{filename}.pdf')

def generate(signal_length, fs, f1):
  time =  np.arange(signal_length)
  signal = 0

  for i in range(1, 5):
    signal += 0.009 * np.cos((2 * np.pi) * (f1 * i)/fs * time)

  plot_spectrogram(signal, fs, 'task6', 'Spektrogram - 4cos.wav')

  wavfile.write('../audio/4cos.wav', fs, signal)

def impulse_response(b, a, i):
  N_imp = 500
  imp = [1, *np.zeros(N_imp-1)] # jednotkovy impuls
  h = signal.lfilter(b, a, imp)

  plt.figure()
  plt.plot(h)
  plt.gca().set_xlabel('$n$')
  plt.gca().set_title('Impulsní odezva $h[n]$')
  plt.savefig(f'task7_{i}.pdf')

def zeros_and_poles(b, a, i):
  z, p, k = signal.tf2zpk(b, a)
  plt.figure(figsize=(7,7))

  # stability
  is_stable = (p.size == 0) or np.all(np.abs(p) < 1) 
  print('Filtr {} stabilní.'.format('je' if is_stable else 'není'))

  # unit circle
  ang = np.linspace(0, 2*np.pi,100)
  plt.plot(np.cos(ang), np.sin(ang))

  # zeros and poles
  plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
  plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

  plt.gca().set_title('Nuly a póly')
  plt.gca().set_xlabel('Reálná složka $\mathbb{R}\{$z$\}$')
  plt.gca().set_ylabel('Imaginární složka $\mathbb{I}\{$z$\}$')

  plt.grid(alpha=0.5, linestyle='--')
  plt.legend(loc='upper left')

  plt.savefig(f'task8_{i}.pdf')


def frequency_response(fs, b, a, i):
  w, H = signal.freqz(b, a)
  _, ax = plt.subplots(1, 2, figsize=(9,3))

  ax[0].plot(w / 2 / np.pi * fs, np.abs(H))
  ax[0].set_xlabel('Frekvence [Hz]')
  ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')

  ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
  ax[1].set_xlabel('Frekvence [Hz]')
  ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

  for ax1 in ax:
      ax1.grid(alpha=0.5, linestyle='--')

  plt.tight_layout()  
  plt.savefig(f'task9_{i}.pdf')

def bandstop_filter(data, fs, f, i):
  nyq = 0.5 * fs
  wp = [(f - 65)/nyq, (f + 65)/nyq]
  ws = [(f - 15)/nyq, (f + 15)/nyq]
  N, Wn = signal.buttord(wp, ws, 3, 40)
  b, a = signal.butter(N, Wn, 'bandstop', output='ba')
  print(f'a{i}:', a)
  print(f'b{i}:', b)

  impulse_response(b, a, i)
  zeros_and_poles(b, a, i)
  frequency_response(fs, b, a, i)

  y = signal.lfilter(b, a, data)

  return y

def filtering(data, fs, f1):
  
  y = data

  for i in range(1, 5):
    y = bandstop_filter(y, fs, f1 * i, i)

  plot_spectrogram(y, fs, 'task10_2', 'Spektrogram - clean_bandstop.wav') 

  wavfile.write('../audio/clean_bandstop.wav', fs, y)

  return y

def main():

  # Task 1
  data, fs = load_file()

  # Task 2
  frame = task_2(data, fs)

  # Task 3
  DFT(frame, fs)

  # Task 4
  plot_spectrogram(data, fs, 'task4', 'Spektrogram - xsvobo1x.wav')

  # Task 5
  # odecteni rusivych frekvenci - rucne
  f1 = 555
  # f2, f3 a f4 jsou násobky f1

  # Task 6
  generate(data.shape[0], fs, f1)

  # Task 7
  #y = bandstop_filter(data, fs, f1)
  #plot_filter(y)
  filtered_data = filtering(data, fs, f1)

  # Task 10
  time = np.linspace(0, len(data)/fs, num=len(filtered_data))
  plot_graph(time, 'Výstupní signál', 'task10_1', 'Čas [s]', 'Amplituda', filtered_data)

main()