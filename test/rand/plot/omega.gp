set terminal svg size 600,400 dynamic enhanced butt solid background "#ffffff"

if (!exists("filename")) filename='plot/omega.dat'
set output filename.".svg"

set xlabel "Killed neurons"
set ylabel "Omega value"
set xtics 1

set style fill transparent solid 0.1 border
set grid noxtics nomxtics noytics nomytics front
set key top left

plot filename using 1:2 with lines lc "#c00000" title "Maximal omega", \
     filename using 1:3 with lines lc "#00c000" title "Average omega"
