#for i in `seq  0 9`
for j in ast
	do
		mkdir images_ETL6/${j}
		#mv ETL6C_${i}_* images_ETL6/${i}/
		# mv ETL6C_${i}_* images_ETL6/${i}/
	done

for i in `seq 0 1382`
	do
		x=19
		# for j in `seq 3 10`
		for j in ast
			do
				x=`expr ${x} + 1`
				mv images_ETL6/tmp/${x}/ETL6C_${x}_${i}.png images_ETL6/${j}/ETL6C_${j}_${i}.png
			done
	done

