# Lucie Svobodova
# header...

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

def plot_graph(time, title, data):
  plt.figure()
  plt.title(title)
  plt.plot(time, data)
  plt.xlabel("Time [s]")
  plt.ylabel("Amplitude")
  #plt.savefig('task_1.pdf')
  plt.show()

def load_file():
  print("Task 1")
  fs, data = wavfile.read('../audio/xsvobo1x.wav')
  # normalise
  data = data / 2**15

  # data info
  print("Sample rate: ", fs)
  print("Length: ", data.shape[0], " samples")
  print("Length: ", data.shape[0]/fs, "sec")
  print("Min: ", data.min())
  print("Max: ", data.max())

  # plot the graph
  time = np.linspace(0, len(data)/fs, num=len(data))
  plot_graph(time, "Task 1", data)

  return fs, data

def task_2(fs, data):
  # ustredneni
  data = data - np.mean(data)

  # normalizace delenim maximem abs
  data_abs = np.absolute(data)
  data = data/max(data_abs)
  print(data)

  time = np.linspace(0, len(data)/fs, num=len(data))
  plot_graph(time, "Task 2 - normalisation", data)

  # rozdeleni signalu na ramce
  frame_len = 1024
  overlap = 512
  frames = [[0] * frame_len for i in range((len(data)//frame_len) * 2 - 1)]
  j = 0
  for i in range(0, len(data) - overlap, overlap):
    frames[j] = data[i : i+frame_len]
    j += 1

  # ulozeni ramcu jako matice - transponovani
  frames = np.array(frames)
  frames = frames.transpose()
  print(frames)

  # zobrazeni pekneho ramce
  frame_num = 24
  nice_guy = frames[0:1024, frame_num]
  print("69", nice_guy)
  
  #time = np.linspace(0, len(data)/fs, num=len(data))
  time = np.linspace(0, len(nice_guy)/fs, num=len(nice_guy))

  plot_graph(time, "Task 2 - frame", nice_guy)

def DFT(fs, data):
  print("DFT")
  print(data) # upravena - jenom jeden ramec tady uz mam


def main():

  # Task 1
  fs, data = load_file()

  # Task 2
  task_2(fs, data)

  # Task 3
  DFT(fs, data)

main()