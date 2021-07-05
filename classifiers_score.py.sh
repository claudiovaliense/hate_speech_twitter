
verifica_processo(){
    while true; do	
        if [ $(( $(pgrep python | wc -l) == 1 )) = 1 ]; then
            sleep 1
        else
            break
        fi
    done
}

varios_datasets(){	
	for c in $2; do
		for r in $3; do
			for dataset in $1; do
				dataset=${dataset}_${r}				
				for index in $(seq 0 $4); do
					verifica_processo				
					MPLBACKEND=Agg python classifiers_score.py ${dataset} dataset/representations/${dataset}/train${index} ${c} ${index} ${5} 				
				done
			done							
		done
	done
}

varios_datasets 'pan21-author-profiling-training-2021-03-142en' 'svm' 'word_tfidf' 0 'f1_macro'
