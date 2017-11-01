for i in `seq 3 12`
	do
		mkdir images_/${i}
		mv ETL3C_${i}_* images/${i}/
	done
