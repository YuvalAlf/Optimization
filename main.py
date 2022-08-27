import time

import cvxpy as cp

from multiplication_matrix import mul_matrix
from poly_coefficients import *


y = cp.Variable(35)
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, \
y26, y27, y28, y29, y30, y31, y32, y33, y34, y35 = y

positive_value = -1 - y1 - 2 * y3 - y5 - 2 * y10 - 2 * y12 - y15 - y35 - 2 * y26 - 2 * y28 - 2 * y31

pos_matrix = cp.reshape(-mul_matrix @ y, (10, 10))

sos_constraint = [pos_matrix >> 0, positive_value >= 0]

poly = y27 * b2011 + y28 * b2020 + y29 * b2101 + y1 * b0004 + y2 * b0013 + y3 * b0022 + y4 * b0031 + y33 * b3010 +\
       y34 * b3100 + y35 * b4000 + y23 * b1201 + y24 * b1210 + y25 * b1300 + y26 * b2002 + y30 * b2110 + y31 * b2200 +\
       y32 * b3001 + y13 * b0301 + y14 * b0310 + y15 * b0400 + y16 * b1003 + y17 * b1012 + y18 * b1021 + y19 * b1030 +\
       y20 * b1102 + y21 * b1111 + y22 * b1120 + y5 * b0040 + y6 * b0103 + y7 * b0112 + y8 * b0121 + y9 * b0130 +\
       y10 * b0202 + y11 * b0211 + y12 * b0220


start_time = time.time()
prob = cp.Problem(cp.Maximize(poly), sos_constraint)
prob.solve()
end_time = time.time()

print("The optimal value is", prob.value)
print("The moments are: ")
print(y.value)
print(f"Time: {(end_time - start_time) * 1000}ms")
