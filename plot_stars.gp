#!/usr/bin/gnuplot 

!shuf -n100000 stars | sort -rnk10 | env LANG=C awk '{$8-=0.91; if($8>3||$8<-1)$8="-"; if($7>3||$7<-1)$7="-"; if($6>3||$6<-1)$6="-"; if($5>3||$5<-1)$5="-"; print;}' > stars.sort
set terminal x11 title "dophot3" persist size 2560,1440

points=0
points=7

set size 1,1
set pointsize 0.5
set yrange [-0.5:0.5]
set cbrange [-1.75:-0.75] #log10(1.091/7)]
unset bars

fit X0 "stars.sort" u 0:3 via X0
fit Y0 "stars.sort" u 0:4 via Y0


list="Catalog Airmass CoordX CoordY Color1 Color2 Color3 Color4"
item(n) = word(list,n)

#set multiplot 
#do for [ i = 0:7 ] {
#    set origin 1./2*(int(i)%2),1./4*int(i/2)
#    set size 0.5,1./4
#    plot "stars.sort" u i+1:(($1-$9)/1):(log10($10))  pt points pal t item(i+1)
#    unset colorbox
#    }
#unset multiplot

set terminal png size 1920,1080
set output "stars-tmp.png"
set multiplot 
do for [ i = 0:7 ] {
    set origin 1./2*(int(i)%2),1./5*int(i/2)
    set size 0.5,1./5
    plot "stars.sort" u i+1:(strcol(16) eq "False"?($1-$9)/1:NaN)  pt points lt rgb "#aaaaaa" t item(i+1),\
     "stars.sort" u i+1:(strcol(16) eq "True"?($1-$9)/1:NaN):(log10($10))  pt points pal t ""
    unset colorbox
    }
i=8
set origin 1./2*(int(i)%2),1./5*int(i/2)
plot "stars.sort" u ($2*$5):(strcol(16) eq "False"?($1-$9)/1:NaN)  pt points lt rgb "#aaaaaa" t "Color x Airmass",\
     "stars.sort" u ($2*$5):(strcol(16) eq "True"?($1-$9)/1:NaN):(log10($10))  pt points pal t ""
i=9
set origin 1./2*(int(i)%2),1./5*int(i/2)
plot "stars.sort" u (sqrt(($4-Y0)**2+($3-X0)**2)):(strcol(16) eq "False"?($1-$9)/1:NaN)  pt points lt rgb "#aaaaaa" t "Radius",\
     "stars.sort" u (sqrt(($4-Y0)**2+($3-X0)**2)):(strcol(16) eq "True"?($1-$9)/1:NaN):(log10($10))  pt points pal t ""
unset multiplot
!rsync stars-tmp.png stars.png
!rm stars-tmp.png

set size 1,1
set pointsize 0.5
set yrange [-1:1]
set cbrange [-1.75:-0.75] #log10(1.091/7)]
unset bars

set terminal png size 1920,1080
set output "stars-ast.png"
set multiplot 
unset colorbox
do for [ i = 0:7 ] {
    set origin 1./2*(int(i)%2),1./5*int(i/2)
    set size 0.5,1./5
    plot \
    "stars.sort" u i+1:(strcol(17) eq "False"?$3-$13:NaN):(log10($10))  pt points lt rgb "#ffaaaa" t "",\
    "stars.sort" u i+1:(strcol(17) eq "False"?$4-$14:NaN):(log10($10))  pt points lt rgb "#aaffaa" t "",\
    "stars.sort" u i+1:(strcol(17) eq "True"?$3-$13:NaN):(log10($10))  pt points lt rgb "#aa0000" t item(i+1),\
    "stars.sort" u i+1:(strcol(17) eq "True"?$4-$14:NaN):(log10($10))  pt points lt rgb "#00aa00" t "",\
    }
i=8
set origin 1./2*(int(i)%2),1./5*int(i/2)
plot \
    "stars.sort" u ($2*$5):(strcol(17) eq "False"?($3-$13)/1:NaN)  pt points lt rgb "#ffaaaa" t "Color x Airmass",\
     "stars.sort" u ($2*$5):(strcol(17) eq "True"?($3-$13)/1:NaN) pt points lt rgb "#aa0000"  t "",\
    "stars.sort" u ($2*$5):(strcol(17) eq "False"?($4-$14)/1:NaN)  pt points lt rgb "#aaffaa" t "",\
     "stars.sort" u ($2*$5):(strcol(17) eq "True"?($4-$14)/1:NaN) pt points lt rgb "#00aa00"  t ""
i=9
set origin 1./2*(int(i)%2),1./5*int(i/2)

plot \
    "stars.sort" u (sqrt(($4-Y0)**2+($3-X0)**2)):(strcol(17) eq "False"?($3-$13)/1:NaN)  pt points lt rgb "#ffaaaa" t "Radius",\
     "stars.sort" u (sqrt(($4-Y0)**2+($3-X0)**2)):(strcol(17) eq "True"?($3-$13)/1:NaN)  pt points lt rgb "#aa0000"  t "",\
    "stars.sort" u (sqrt(($4-Y0)**2+($3-X0)**2)):(strcol(17) eq "False"?($4-$14)/1:NaN)  pt points lt rgb "#aaffaa" t "",\
     "stars.sort" u (sqrt(($4-Y0)**2+($3-X0)**2)):(strcol(17) eq "True"?($4-$14)/1:NaN)  pt points lt rgb "#00aa00"  t ""
unset multiplot

exit

set yrange [-0.2:0.2]

set terminal png size 1920,1080
set output "stars-c.png"
set multiplot 
do for [ i = 4:7 ] {
    set origin 1./2*(int(i)%2),1./2*int(i/2)-1
    set size 0.5,1./2
    plot [-1:3] "stars.sort" u i+1:(($1-$9)/1):(log10($10))  pt points pal t item(i+1)
    unset colorbox
    }
unset multiplot

in 3d:
splot [][][-0.5:0.5] "stars.sort" u ($4-Y0):($3-X0):(strcol(17) eq "False"?($1-$9)/1:NaN), "stars.sort" u ($4-Y0):($3-X0):(strcol(17) eq "True"?($1-$9)
plot "stars.sort" u ($4-Y0):($3-X0):(strcol(17) eq "False"?($4-$14)/1:NaN), "stars.sort" u ($4-Y0):($3-X0):(strcol(17) eq "True"?($4-$14)/1:NaN)
/1:NaN)

