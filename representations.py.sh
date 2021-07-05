#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Execute create representations of texts

verifica_processo(){ # gerencia o numero de execucoes que o computador suporta
    while true; do	
        if [ $(( $(pgrep python | wc -l) == 5 )) = 1 ]; then
            sleep 1
        else
            break
        fi
    done
}

varios_datasets(){
    for r in $3; do
        for dataset in $1; do					
            for index in $(seq 0 $2); do
                verifica_processo
                MPLBACKEND=Agg python representations.py ${dataset}_${r} dataset/$dataset/split_5.csv dataset/$dataset/orig/texts.txt dataset/$dataset/orig/score.txt $index ${r} & 
            done		
        done
    done
}

# exemplos de execucoes
varios_datasets 'pan21-author-profiling-training-2021-03-142en' 4 'word_tfidf' 
#varios_datasets 'pan21-author-profiling-training-2021-03-142en' 4 'vader' 
#varios_datasets 'pan21-author-profiling-training-2021-03-142en' 4 'char_tfidf' 
#varios_datasets 'pan21-author-profiling-training-2021-03-142en' 4 'roberta_concat' 

