for folder in /mnt/data/covid_our_data_v3/data/* 
do
	echo $folder 
	for i in $folder/*.dcm
	do
		echo $i
		lungmask $i $i
	#echo $i | awk -F/ '{print $NF}'
	done
done
