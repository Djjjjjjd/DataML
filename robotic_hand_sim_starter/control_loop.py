import numpy as np
import matplotlib.pyplot as plt

# --- Параметры ---
fs = 1000
duration = 6.0
t = np.arange(0, duration, 1/fs)

# Синтетический ЭМГ: две активации разной силы
np.random.seed(42)
noise = 0.1 * np.random.randn(len(t))
act = np.zeros_like(t)
act[(t > 1.0) & (t < 2.2)] = 0.7
act[(t > 3.2) & (t < 5.0)] = 1.0
carrier = np.random.randn(len(t))
emg = (0.1 + act) * carrier + noise

# --- Фильтры ---
# Абсолютное значение
abs_emg = np.abs(emg)
# Скользящее среднее (20 мс)
win = int(0.02 * fs)
win = max(win, 1)
kernel = np.ones(win) / win
env = np.convolve(abs_emg, kernel, mode="same")

# --- Порог + гистерезис ---
th_on = np.percentile(env, 70)   # включение
th_off = np.percentile(env, 40)  # выключение
state = 0  # 0=open, 1=close
state_trace = np.zeros_like(env)
for i, v in enumerate(env):
    if state == 0 and v > th_on:
        state = 1
    elif state == 1 and v < th_off:
        state = 0
    state_trace[i] = state

# --- "Сервопривод": плавная подводка к целевому углу ---
angle = 0.0
angles = np.zeros_like(t)
speed_deg_per_s = 240.0  # ограничение скорости
dt = 1.0/fs
for i in range(len(t)):
    target = 160.0 if state_trace[i] > 0.5 else 10.0
    # Плавная подводка
    step = speed_deg_per_s * dt
    if angle < target:
        angle = min(angle + step, target)
    elif angle > target:
        angle = max(angle - step, target)
    angles[i] = angle

# --- Графики ---
plt.figure()
plt.title("EMG Envelope and Hysteresis States")
plt.plot(t, env, label="Envelope")
plt.axhline(th_on, linestyle="--", label="th_on")
plt.axhline(th_off, linestyle="--", label="th_off")
plt.plot(t, state_trace * env.max(), label="State (scaled)")
plt.xlabel("Time, s")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.title("Simulated Servo Angle")
plt.plot(t, angles, label="Servo angle (deg)")
plt.xlabel("Time, s")
plt.ylabel("Degrees")
plt.legend()
plt.tight_layout()
plt.show()
