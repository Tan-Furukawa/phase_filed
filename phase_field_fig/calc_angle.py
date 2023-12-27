#%%%
import numpy as  np

#     C          D
#     /----------/
#    /          /
#   /          / b
#  /      beta/
# /----------/
#A     a     B
# estimate /_CBA
#
beta = 116 * np.pi / 180
a = 1 * 8.30
b = 6 * 7.14
sinCBA = np.sin(beta) / np.sqrt(1 + (a/b)**2 - 2 * (a/b)*np.cos(np.pi - beta))
angle = np.arcsin(sinCBA)
print(90 - angle * 180 / np.pi + 90)
