import thick
import t_OX
import scaling
import rot
import t_OY
import TPH
import TDH
import timeit

tOx = timeit([t_OX.estim_X(i) for i in range(2000)], number = 1)
tOy = timeit([t_OY.estim_Y(i) for i in range(2000)], number = 1)
trot = timeit([rot.estim_R(i) for i in range(2000)], number = 1)
tscal =timeit([scaling.estim_S(i) for i in range(2000)], number = 1)
ttph = timeit([TPH.estim_TPH(i) for i in range(2000)], number = 1)
ttdh = timeit([TDH.estim_TDH(i) for i in range(2000)], number = 1)
tthick = timeit([thick.estim_T(i) for i in range(2000)], number = 1)


with open('timings.txt', 'w') as filehandle:
    filehandle.write('%s\n Ox:' % tOx)
    filehandle.write('%s\n Oy:' % tOy)
    filehandle.write('%s\n rot:' % trot)
    filehandle.write('%s\n scaling:' % tscal)
    filehandle.write('%s\n tph:' % ttph)
    filehandle.write('%s\n tdh:' % ttdh)
    filehandle.write('%s\n thick:' %tthick)
