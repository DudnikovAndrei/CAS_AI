import numpy as np
import matplotlib.pyplot as plt

# Parameters for discretizing the mathematical function
sampling = 100
x_range = -10, 10
n_waves = 2

#Parameters are tuples with a value for each wave (2 in this case)
amplitudes = 1.7, 0.8
wavelengths = 4, 7.5
velocitys = 4, 7.5
time = 0

x = np.linspace(x_range[0], x_range[1], sampling)

# Create 2 (or more) waves using a list comprehension and superimpose
waves = [amplitudes[idx] * np.sin((2*np.pi/wavelengths[idx]) * (x - velocitys[idx]*time)) for idx in range(n_waves)]
superimposed_wave = sum(waves)

plt.subplot(2, 1, 1)
plt.plot(x, waves[0])
plt.plot(x, waves[1])

plt.subplot(2, 1, 2)
plt.plot(x, superimposed_wave)

plt.show()