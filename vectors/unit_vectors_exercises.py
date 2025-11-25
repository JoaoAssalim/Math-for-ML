# Encontre a forma retangular de v + w
# (r, ø) = magnetude e angulo
# v = (6, 100º)
# w = (5, 40º)

import math

x = 8 * math.cos(math.radians(100)) + 3 * math.cos(math.radians(210))
y = 8 * math.sin(math.radians(100)) + 3 * math.sin(math.radians(210))

print(round(x, 2), round(y, 2))