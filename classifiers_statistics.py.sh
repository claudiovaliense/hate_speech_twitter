varios_datasets(){
    # varios_dataset dataset representacao folds classificador
	for c in $4; do
		for r in $2; do
		    for dataset in $1; do
				#python3.6 classifiers_statistics.py ${dataset} 5 bert f1_macro & #bert
				MPLBACKEND=Agg python classifiers_statistics.py ${dataset}_${r} $3 ${c} $5
		    done
		done
	done
}


varios_datasets 'pan21-author-profiling-training-2021-03-142en' 'word_tfidf' 5 'svm' f1_macro
