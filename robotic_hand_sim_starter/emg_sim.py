import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

fs = 1000  # Гц
t = np.arange(0, 5.0, 1/fs)  # 5 секунд
# Базовый шум (покой)
noise = 0.1 * np.random.randn(len(t))

# Периоды активации мышцы (прямоугольные окна)
act = np.zeros_like(t)
act[(t > 1.0) & (t < 2.0)] = 1.0
act[(t > 3.0) & (t < 4.0)] = 1.0

# ЭМГ как амплитудно-модулированный шум
carrier = np.random.randn(len(t))
emg = (0.15 + 0.85 * act) * carrier + noise

# Простая огибающая: |x| -> скользящее среднее
abs_emg = np.abs(emg)
win = int(0.02 * fs)  # 20 мс окно
win = max(win, 1)
kernel = np.ones(win) / win
env = np.convolve(abs_emg, kernel, mode="same")

plt.figure()
plt.title("Synthetic EMG and Envelope")
plt.plot(t, emg, label="EMG")
plt.plot(t, env, label="Envelope")
plt.plot(t, act, label="Ground Truth (activation)")
plt.xlabel("Time, s")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
