set terminal svg size 600,400 dynamic enhanced butt solid background "#ffffff"

if (!exists("filename")) filename='plot/transfert.dat'
set output filename.".svg"

set xlabel "Input"
set ylabel "Output"

set style fill transparent solid 0.1 border
set grid noxtics nomxtics noytics nomytics front
set key top left

plot filename using 1:2 with lines lc "#c00000" title "Transfert function", \
     filename using 1:3 with lines lc "#00c000" title "Derivative function"
