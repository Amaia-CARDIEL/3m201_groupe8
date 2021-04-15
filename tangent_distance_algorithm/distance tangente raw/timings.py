import thick
import t_OX
import scaling
import rot
import t_OY
import TPH
import TDH
import timeit

print(timeit([t_OX.estim_X(i) for i in range(2000)], number = 1))
print(timeit([t_OY.estim_Y(i) for i in range(2000)], number = 1))
print(timeit([rot.estim_R(i) for i in range(2000)], number = 1))
print(timeit([scaling.estim_S(i) for i in range(2000)], number = 1))
print(timeit([TPH.estim_TPH(i) for i in range(2000)], number = 1))
print(timeit([TDH.estim_TDH(i) for i in range(2000)], number = 1))
print(timeit([thick.estim_T(i) for i in range(2000)], number = 1))
