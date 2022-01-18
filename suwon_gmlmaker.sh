#!/bin/bash
#SBATCH -J java
#SBATCH -p ivy_v100-16G_2
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -e %j.stderr
#SBATCH -o %j.stdout
#SBATCH --time=2:00:00
#SBATCH --comment=etc
unset DISPLAY
java -Djava.awt.headless=true -cp /scratch/x2026a02/new_rcrs/rcrs-server/jars/*:/scratch/x2026a02/new_rcrs/rcrs-server/lib/* maps.convert.Convert "/scratch/x2026a02/new_rcrs/rcrs-server/suwon_big/map.osm" "/scratch/x2026a02/new_rcrs/rcrs-server/suwon_big/map.gml"
