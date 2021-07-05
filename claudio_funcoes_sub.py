#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Create general purpose library
import unidecode # remove accents
import io
import json  # Manipulate extension json
import collections # quantifica elementos em listas e conjuntos
import random
from datetime import datetime  # Datetime for time in file
import scipy.stats as stats # Calcular intervalo de confiança
import os  # Variable in system
import ast # String in dict
import operator # intevalo de confianca teste nao parametrico
from scipy.stats import norm
import re # Regular expression
import timeit  # calcular metrica de tempo
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from heapq import nlargest  # Order vector]
import statistics
import numpy
from sklearn.metrics import confusion_matrix
import pandas as pd # matrix confusion

def str_latex(string):
    """Replace invalid caracters in latex. Example: str_latex('svm_tfidf'); Return='svm\\_tfidf'"""
    return string.replace("_", "\\_")   

def matrix_confuion(y_true, y_pred, y_labels, columns_labels):
    """Return confusion matrix with title. Example: matrix_confuion([1,0,1], [1,1,0], [0,1], ['neg', 'pos']); Return=matrix """
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=y_labels),
        index=columns_labels,
        columns=columns_labels
    )
    return matrix.to_string() 

def matrix_confusion_folds(y_test_folds, y_pred_folds, y_labels, columns_labels):
    """List matrix confunsion per fold. Example: matrix_confuion_folds([[1,0,1]], [[1,1,0]], [0,1], ['neg', 'pos']); Return=String"""
    out = "y_pred = coluna, y_true = linhas\n"

    for index in range(len(y_test_folds)):
        out = str_append(out, f'FOLD {index}')
        out = str_append(out, matrix_confuion(y_test_folds[index], y_pred_folds[index], y_labels, columns_labels))

        #print(f"FOLD {index}:") 
        #print(matrix_confuion(y_test_folds[index], y_pred_folds[index], y_labels, columns_labels))
        #print(matrix_confuion(y_test_folds[index], y_pred_folds[index], [-1,1], ['positive', 'negative']))
    return out

def str_append(string, add):
    """Append add in end string. Example: str_append('hou', 'se'); Return='house'"""
    return string + str(add) + "\n"    

def accuracy_folds(y_test_folds, y_pred_folds):
    """ Accuracy score of the various lists. Example: accuracy_folds([[1,0,1]], [[1,1,1]]); Return=0.66 """
    metric=[]
    for index in range(len(y_test_folds)):
        metric.append(sklearn.metrics.accuracy_score(y_test_folds[index], y_pred_folds[index]))
    return metric

def name_out(file_csv):
    """ Return name of out file with datetime. Example: function('my_file.txt')"""
    name = os.path.basename(file_csv)
    file_name = os.path.splitext(name)[0]
    file_type = os.path.splitext(name)[1]
    file_location = os.path.dirname(file_csv) + "/"
    date = "_" + datetime.now().strftime('%d-%m-%Y.%H-%M-%S')
    return file_location + file_name + date + file_type

def f1(y_test_folds, y_pred_folds, average='macro'):
    """Return f1 score of the various lists. Example: f1([[1,0,1]], [[1,1,1]], 'macro'); Return=0.4"""
    metric=[]
    for index in range(len(y_test_folds)):    
        metric.append(sklearn.metrics.f1_score(y_test_folds[index], y_pred_folds[index], average=average))
    return metric

def statistics_experiment(name_dataset, classifier, y_pred_folds, y_test_folds, best_param_folds, time_method, metric, folds):
    """Write file and print statistics experiment. Example: statistics_experiment('debate', 'svm', [[1,0,1]], [[1,1,0]], [{C:'100'}], [1,2,3], 'f1_macro', 1) """     
    # encoding = "ISO-8859-1"    
    
    print( f1(y_test_folds, y_pred_folds, 'macro') ) # teste 1 fold
    time_representation = []
    time_predict = []
    time_train = []
    for i in range(folds):
        time_representation.append(0) #bert
        #time_representation.append(load_dict_file("times/"+name_dataset+"_"+str(i))['time_representation']) #comentar para bert        
        time_predict.append(load_dict_file("y_pred/"+name_dataset+"_time_"+str(i))['time_predict'])
        time_train.append(load_dict_file("y_pred/"+name_dataset+"_time_"+str(i))['time_train'])

    result_grid = []
    escores=[]
    time_grid_times = []
    #metric = 'f1_macro'
    if classifier != 'bert':        
        #for index_fold in range(folds): # load os escore do grid
        for index_fold in range(1): # apenas o escore do primeiro fold
            escore = []
            escores.append( load_dict_file('escores/' +name_dataset  +'_' +classifier +'_escore_grid_train' +str(index_fold) ) )
            index_best = rank_grid(escores[index_fold][metric],1)[0]
            result_grid.append( escores[index_fold][metric][index_best]['score'] ) # resultado do fold de cada grid


            time_grid = load_dict_file("escores/"+name_dataset +"_"+classifier +"_escore_grid_train"+str(index_fold))['time_grid']
            time_grid_times.append(time_grid)
    else:
        result_grid.append(0)
        result_grid.append(0)
        time_grid = {'time_grid': 0}
    

    
    with open(name_out('./statistics/'+name_dataset+"_"+classifier), 'w', errors='ignore') as file_write:
        out = ""
        f1_macro = f1(y_test_folds, y_pred_folds, 'macro')
        #print('macro', f1_macro)
        f1_micro = f1(y_test_folds, y_pred_folds, 'micro')
        accuracy = accuracy_folds(y_test_folds, y_pred_folds)
        #roc_auc = roc_auc_folds(y_test_folds, y_pred_folds) # dataset covid
        out = str_append(out, "Name Dataset: " + str(name_dataset))
        out = str_append(out, "Best_param_folds: " + str(best_param_folds))
        out = str_append(out, "Macro F1 por fold: " + str(f1_macro))
        out = str_append(out, "Micro F1 por fold: " + str(f1_micro))
        out = str_append(out, 'Média Macro F1: ' + str(statistics.mean(f1_macro)))
        out = str_append(out, "Desvio padrão Macro F1: " + str(statistics.stdev(f1_macro)))
        out = str_append(out, 'Média Micro F1:  ' + str(statistics.mean(f1_micro)))
        out = str_append(out, "Desvio padrão Micro F1: " + str(statistics.stdev(f1_micro)))  
        out = str_append(out, "Time method: " + str(max(time_method)))
       
        from sklearn.metrics import  precision_recall_fscore_support #metric per class
        p_r_f_s = [ precision_recall_fscore_support(y_test_folds[index], y_pred_folds[index]) for index in range(folds) ] 
        p = []; r = []; f = []; s = []
        for fold_metrics in p_r_f_s:
            p.append(fold_metrics[0].tolist()); r.append(fold_metrics[1].tolist()); f.append(fold_metrics[2].tolist()); s.append(fold_metrics[3].tolist())

        from sklearn.metrics import classification_report
        print(classification_report(y_test_folds[0], y_pred_folds[0]))

        ''' # saude covid
        roc_list = []
        for index_fold_prob in range(folds):            
            y_prob = load_dict_file(f'y_prob/covid_ufmg_svmlight_{classifier}_f1_macro_test{index_fold_prob}')['y_pred-folds']['0']
            y_prob = numpy.array(y_prob)            
            roc_list.append( roc_auc_score( y_test_folds[index_fold_prob], y_prob[:,1] )  ) 
        print(statistics.mean(roc_list))
        '''

        

        dict_statistics = {
            'name_dataset' : name_dataset,
            'best_param_folds' : str(best_param_folds),
            'precision_class' : p,
            'recall_class' : r,
            'f1_class' : f,
            'suport_class' : s,
            'macro_f1' : f1_macro,
            'micro_f1' : f1_micro,
            'mean_macro_f1' : statistics.mean(f1_macro),
            'std_macro_f1' : statistics.stdev(f1_macro),
            'mean_micro_f1' : statistics.mean(f1_micro),
            'std_micro_f1' : statistics.stdev(f1_micro),
            'time_representation' : time_representation,
            'mean_time_representation' : statistics.mean(time_representation),
            'time_train' : time_train,
            'mean_time_train': statistics.mean(time_train),
            'time_predict': time_predict,
            'mean_time_predict': statistics.mean(time_predict),
            'time_grid' : time_grid_times,
            'grid_macro_f1' : result_grid,
            'accuracy' : accuracy#,
            #'roc_auc' : roc_list # dataset covid
        }
        
        save_dict_file('statistics/'+name_dataset +"_" +classifier, dict_statistics)
        #save_dict_file('statistics/'+name_dataset+"_"+"bert", dict_statistics) #bert

       # print(time_grid['time_grid'])
        times_method = []
        for index_time in range(folds):
            time_method = time_representation[index_time] +time_grid_times[0] + time_train[index_time] + time_predict[index_time] # apenas escore do primeiro fold
            #time_method = time_representation[index_time] +time_grid_times[index_time] + time_train[index_time] + time_predict[index_time]
            times_method.append(time_method)

        classifier = classifier.replace("_", "\\_")
        out = str_append(out, matrix_confusion_folds(y_test_folds, y_pred_folds, numpy.unique(y_test_folds[0]), numpy.unique(y_test_folds[0])))
        print(out)
        if metric == 'f1_macro':
            #print(graph_latex(f1_macro, max(time_method), classifier)) 
            format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier +" & " +str('%.2f'%(statistics.mean(f1_macro)*100)) +"$\pm$" +str('%.2f'%( ic( len(f1_macro), statistics.stdev(f1_macro), 0.95, type='t')  *100)) +"\\\\" # +"&" +str('%.2f'%(statistics.mean( times_method ))) +' (' +str('%.2f'%( ic(len(times_method), statistics.stdev(times_method), 0.95, type='t')))    +") \\\\"
        elif metric == 'f1_micro':
            #print(graph_latex(f1_micro, max(time_method), classifier) )
            format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier +" & " +str('%.2f'%(statistics.mean(f1_micro)*100)) +"$\pm$" +str('%.2f'%( ic( len(f1_micro), statistics.stdev(f1_micro), 0.95, type='t')  *100)) +"\\\\"             
        elif metric == 'accuracy':
            format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier +" & " +str('%.2f'%(statistics.mean(accuracy)*100)) +"$\pm$" +str('%.2f'%( ic( len(accuracy), statistics.stdev(accuracy), 0.95, type='t')  *100)) +"\\\\" # +"&" +str('%.2f'%(statistics.mean( times_method ))) +' (' +str('%.2f'%( ic(len(times_method), statistics.stdev(times_method), 0.95, type='t')))    +") \\\\"
            #format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier +" & " +str('%.2f'%(statistics.mean(accuracy)*100)) +" (" +str('%.2f'%(statistics.stdev(accuracy)*100)) +") & " +str('%.2f'%(max(time_method)))   +" \\\\"
        #elif metric == 'roc_auc':
            #print('aslkasjska', y_pred_folds[0])            
            #print(sklearn.metrics.roc_auc_score(y_test_folds[0], y_pred_folds[0], average='micro') )
            #format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier +" & " +str('%.2f'%(statistics.mean(roc_auc)*100)) +"$\pm$" +str('%.2f'%( ic( len(roc_auc), statistics.stdev(accuracy), 0.95, type='t')  *100)) +"\\\\" # +"&" +str('%.2f'%(statistics.mean( times_method ))) +' (' +str('%.2f'%( ic(len(times_method), statistics.stdev(times_method), 0.95, type='t')))    +") \\\\"
            #'''
        
            
            '''y_prob_index = numpy.argmax(y_prob, axis=1)
            y_prob_temp = []
            for index_prob in range( len( y_prob_index)):
                if  y_prob_index[index_prob] == 1:
                    y_prob_temp.append( y_prob[index_prob ] [ y_prob_index[index_prob] ]  )
                else:
                    y_prob_temp.append( y_prob[index_prob ] [ y_prob_index[index_prob] ] / 2  )
            '''
            #y_prob = y_prob_temp
            #print(y_prob)
            #y_test = [ int(y) for y in y_test_folds[0]]    
            #y_pred = [ int(y) for y in y_pred_folds[0]]
            #print( y_test)                    
            #fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_test), np.array(y_prob), pos_label=1)  
            #print( sklearn.metrics.auc(fpr, tpr)); exit()
            #fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_test), np.array(y_pred), pos_label=1)                        
            #print(fpr)
            #format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier +" & " +str(sklearn.metrics.auc(fpr, tpr)) +" \\\\"
 

        out = str_append(out, format_latex)
        file_write.write(out)   
    #y_prob = load_dict_file('y_prob/covid_ufmg_svmlight_random_forest_roc_auc_test0')['y_pred-folds']['0'] 
    #y_prob_temp = []
    #for y in y_pred_folds[0]:
    #    y_prob_temp.append( y[1])
    #print( sklearn.metrics.roc_auc_score(y_test_folds[0],y_prob_temp)); exit()
    print(format_latex)

def save_dict_list(y_true_folds, y_pred_folds, filename):
    """Save predict and true in file. Example: save_dict_list([[1,0,1],[1,1,1]], [[1,1,0],[1,0,1]], 'myfile.json'); Return=Save File"""
    y_pred = dict()
    y_true = dict()
    y=dict()
    for index in range(len(y_pred_folds)):
        y_pred[index]=y_pred_folds[index].tolist()
        #y_true[index]=y_true_folds[index]
        y_true[index]=y_true_folds[index].tolist()
    y['y_pred-folds'] = y_pred
    y['y_true-folds'] = y_true
    save_dict_file(filename, y)


def k_max_index(list, k):
    """ Return index of max values. Example: k_max_index([2, 4, 5, 1, 8], 2)
    Example:
    r = [0.5, 0.7, 0.3, 0.3, 0.3, 0.4, 0.5]
    print(k_max_index(r, 3))
    """''
    list_m = list.copy()
    max_index = []
    k_max_values = nlargest(k, list_m)
    for k_value in k_max_values:
        index_k = list_m.index(k_value)
        max_index.append(index_k)
        list_m[index_k] = -1
    return max_index

def rank_grid(rank, k_max):
    """Rank considering standard deviation. Example: rank_grid([ {'score': 0.4}, {'score': 0.5} ], 1). Return=[1] """  
    rank_scores=[]  
    for itens in rank:
        #rank_scores.append(itens['score']-itens['std'])  
        rank_scores.append(itens['score'])#-itens['std'])  
                
    return k_max_index(rank_scores,k_max)  

def best_param_folds_no_frequency(escores, index_fold, metric):
    """Return best param in fold. Example: best_param_folds_no_frequency([ {'score': 0.4}, {'score': 0.5} ], 0, 'macro_f1')"""
    index_best = rank_grid(escores[index_fold][metric],1)[0]
    #print(escores[index_fold][metric][index_best])
    return escores[index_fold][metric][index_best]['params']

def load_dict_file(file):
    """Load dict in file. Example: load_dict_file('myfile.json')"""
    try:
        with open(file, 'r', newline='') as csv_reader:
            return json.load(csv_reader)
    except OSError:
        return {}

def load_escores(name_dataset, classifier, folds):
    """Excluir xxxxxxx Return escore in fold. """
    escores=[]
    escores.append(load_dict_file("escores/"+name_dataset +"_"+classifier +"_escore_grid_train"+str(folds)))
    return escores
    for index in range(folds):
        escores.append(load_dict_file("escores/"+name_dataset +"_"+classifier +"_escore_grid_train"+str(index)))
    return escores


def load_dict_list(filename):
    """Return list formado por um dict. Example: load_dict_list('filename.json'). Return=[value_dict, value_dict, ...]"""
    my_dict = load_dict_file(filename)
    lista = []
    for i in range(len(my_dict.keys())):
        lista.append( my_dict[str(i)] )
    return lista

def escores_grid(name_dataset, classifier, name_train, estimator, tuned_parameters, metrics, refit):
    """ Return escores grid. Example: escores_grid('debate', 'svm', 'train0', {'C':[1, 10, 100]}, ['f1_macro'], 'f1_macro'); Return=save escores in file """    
    escores=dict()    
    #metrics = ['f1_macro']
    parallel = -1 #-1, 4
    if classifier != 'bert':
        x_train, y_train = load_svmlight_files([open(name_train, 'rb')])
    else:
        x_train = load_dict_list(name_train)
        y_train = load_dict_list(name_train +"_label")
        parallel = 1

    if classifier == 'knn':
        parallel = 8
   
    #x_train = [x_train[index].toarray()[0] for index in range(x_train.shape[0])]
    #estimator.fit(x_train,y_train) # baixar bert e testa
    #exit()
    #scaler = MinMaxScaler()
    #x_train = scaler.fit_transform(x_train)
    #if y == "1"
    #for index_y in range( len( y_train) ):
    #    if y_train[index_y] == -1: y_train[index_y] = 0 

    grid = GridSearchCV(estimator, param_grid=tuned_parameters,  cv=5, scoring=metrics[0], n_jobs=parallel)#, refit=refit)#, verbose=30)
    ini = timeit.default_timer()
    grid.fit(x_train, y_train)  
    escores['time_grid'] = timeit.default_timer() - ini
    for metric_value in metrics:
        escores_list=[]
        #means = grid.cv_results_['mean_test_'+metric_value]
        #stds = grid.cv_results_['std_test_'+metric_value]
        means = grid.cv_results_['mean_test_score']                
        stds = grid.cv_results_['std_test_score']
        time_params = grid.cv_results_['mean_fit_time']
        for mean, std, params, time_param in zip(means, stds, grid.cv_results_['params'], time_params):
            escores_list.append({'score': mean, 'std': std, 'params': params, 'time_param' : time_param})        
        escores[metric_value] = escores_list
    
    file = "escores/"+name_dataset +"_"+classifier +"_escore_grid_"+name_file(name_train)
    
    '''
    try: # adicionar escore a um escore existente
        with open(file, 'r') as file_reader: # append params in dict
            old_escore = json.load(file_reader)
            
            for metric_value in metrics:
                if old_escore.get(metric_value) == None:
                    old_escore[metric_value] = escores[metric_value]
                    escores = old_escore
                else:                    
                    escore_list_old = old_escore[metric_value]
                    for params in escores[metric_value]: # add new escores in old escores
                        escore_list_old.append(params)
                    escores[metric_value] = escore_list_old                
    except IOError: 
        pass              
    '''

    #import claudio_funcoes as cv
    #print(cv.best_param_folds_no_frequency([escores], 0, 'f1_macro'))
    save_dict_file(file, escores)
    #joblib.dump(grid, "escores/" + 'grid_'+name_dataset+"_"+classifier)
    #json.dumps(grid.cv_results_, default=default)

      
    #save_dict_file("escores/"+name_dataset +"_svm_escore_grid_"+name_file(name_train), grid.cv_results_)  

    #return grid

def ic(tamanho, std, confianca, type='t', lado=2):
    """Calcula o intervalo de confianca"""
    if lado is 1:
        lado = (1 - confianca) # um lado o intervalo fica mais estreito
    else:
        lado = (1 - confianca) /2 
        
    #print(f'Valor de t: {stats.t.ppf(1- (lado), tamanho-1) }')    
    if type is 'normal':
        return stats.norm.ppf(1 - (lado)) * ( std / ( tamanho ** (1/2) ) )
    return stats.t.ppf(1- (lado), tamanho-1) * ( std / ( tamanho ** (1/2) ) ) 

def save_dict_file(file, dict):
    """Save dict in file. Example: save_dict_file('myfile.json', {'1' : 'hello world'})"""
    with open(file, 'w', newline='') as json_write:
        json.dump(dict, json_write)


def ids_train_test(ids_file, datas, labels, id_fold):
    """Return data and labels starting of ids file. Example: ids_train_test('ids.txt', 'texts.txt', 'labels.txt', 0); Return=x_train, y_train, x_test, y_test"""
    ids = file_to_corpus(ids_file)
    train_test = str(ids[id_fold]).split(';')
    ids_train = [int(id) for id in train_test[0].strip().split(' ')]
    ids_test = [int(id) for id in train_test[1].strip().split(' ')]
    total = file_to_corpus(datas)
    labels = file_to_corpus(labels)
    x_train = [total[index] for index in ids_train]       
    y_train = [int(labels[index]) for index in ids_train] 
    x_test = [total[index] for index in ids_test]
    y_test = [int(labels[index]) for index in ids_test] 
    return x_train, y_train, x_test, y_test

def ids_train_test_shuffle(ids_file, datas, labels, id_fold):    
    """Return data and labels starting of ids file. Example: ids_train_test('ids.txt', 'texts.txt', 'labels.txt', 0); Return=x_train, y_train, x_test, y_test"""
    ids = file_to_corpus(ids_file)
    train_test = str(ids[id_fold]).split(';')
    ids_train = [int(id) for id in train_test[0].strip().split(' ')]
    ids_test = [int(id) for id in train_test[1].strip().split(' ')]
    new_ids_train = ids_train.copy(); new_ids_test = ids_test.copy()
    random.shuffle(new_ids_train); random.shuffle(new_ids_test)
    total = file_to_corpus(datas)
    labels = file_to_corpus(labels)
    x_train = [total[index] for index in new_ids_train]       
    y_train = [int(labels[index]) for index in new_ids_train] 
    x_test = [total[index] for index in new_ids_test]
    y_test = [int(labels[index]) for index in new_ids_test] 
    return x_train, y_train, x_test, y_test, ids_train, ids_test, new_ids_train, new_ids_test

def ids_train_test_shuffle(ids_file, datas, labels, id_fold):
    """Return data and labels starting of ids file. Example: ids_train_test('ids.txt', 'texts.txt', 'labels.txt', 0); Return=x_train, y_train, x_test, y_test"""
    ids = file_to_corpus(ids_file)
    train_test = str(ids[id_fold]).split(';')
    ids_train = [int(id) for id in train_test[0].strip().split(' ')]
    ids_test = [int(id) for id in train_test[1].strip().split(' ')]
    new_ids_train = ids_train.copy(); new_ids_test = ids_test.copy()
    random.shuffle(new_ids_train); random.shuffle(new_ids_test)
    total = file_to_corpus(datas)
    labels = file_to_corpus(labels)
    x_train = [total[index] for index in new_ids_train]       
    y_train = [int(labels[index]) for index in new_ids_train] 
    x_test = [total[index] for index in new_ids_test]
    y_test = [int(labels[index]) for index in new_ids_test] 
    return x_train, y_train, x_test, y_test, ids_train, ids_test, new_ids_train, new_ids_test

def file_to_corpus(name_file):
    """Transforma as linahs de um arquivo em uma lista. Example: function(my_file.txt) """
    rows = []
    #with open(name_file, 'r') as read:
    #, encoding = "ISO-8859-1" , encoding='utf-8'
    with io.open(name_file, newline='\n', errors='ignore') as read: #erro ignore caracteres
        for row in read:
            row = row.replace("\n", "")
            rows.append(row)
    return rows


def remove_accents(string):
    """ Remove accents string. """
    return unidecode.unidecode(string)

def remove_point_virgula(text):
    text =  re.sub("[.,]", " ", text)
    return re.sub(' +', ' ', text) # remove multiple space

def preprocessor(text):
    """ Preprocessoing data. Example: cv.preprocessor('a155a 45638-000'); Return='a ParsedDigits a Parsed-ZopcodePlusFour'"""        
    replace_patterns = [
    ('<[^>]*>', 'parsedhtml') # remove HTML tags       
    ,(r'(\D)\d\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d\d\d\-\d\d\d\d(\D)', 'ParsedPhoneNum') # text_phone_text
    ,(r'(\D)\d\d\d\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text
    ,(r'(\D\D)\d\d\d\D\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text
    ,(r'(\D)\d\d\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedZipcodePlusFour \\2') # text_zip_text
    ,(r'(\D)\d\d\d\d-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text  
    ,(r'\d\d:\d\d:\d\d', 'ParsedTime') #time
    ,(r'\d:\d\d:\d\d', 'ParsedTime') #time
    ,(r'\d\d:\d\d', 'ParsedTime') # time
    ,(r'\d\d\d-\d\d\d\d', 'ParsedPhoneNum') # phone US
    ,(r'\d\d\d\d-\d\d\d\d', 'ParsedPhoneNum') # phone brasil
    ,(r'\d\d\d\d\d\-\d\d\d\d', 'ParsedZipcodePlusFour') # zip
    ,(r'\d\d\d\d\d\-\d\d\d', 'ParsedZipcodePlusFour') # zip brasil
    ,(r'(\D)\d+(\D)', '\\1 ParsedDigits \\2') # text_digit_text
    ]    
    compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]
    
    # For each pattern, replace it with the appropriate string
    for pattern, replace in compiled_replace_patterns:        
        text = re.sub(pattern, replace, text)
    
    #text = remove_caracters_especiais_por_espaco(text)
    text = remove_accents(text)
    text = text.lower()        
    text = remove_point_virgula(text)
    
    text = text.split(" ")
    index=0
    for t in text:
        if text[index].__contains__("http://"):
            text[index] = 'parsedhttp'
        elif text[index].__contains__("@"):
            text[index] = 'parsedref'
        index+=1
    return " ".join(text)

def name_file(file_csv):
    """ Return name of out file. Example: function('Downloads/myfile.txt')"""
    name = os.path.basename(file_csv)
    file_name = os.path.splitext(name)[0]    
    return file_name


def ids_train_test_representation(ids_file, datas, id_fold, sep_index):
    """Return dados da representacao correspondente ao fold. Example: ids_train_test_representation('ids.txt', 'tests.txt', 0, ' '); Return=[ids_train],[ids_test]"""
    ids = file_to_corpus(ids_file)
    train_test = str(ids[id_fold]).split(';')
    ids_train = [int(id) for id in train_test[0].strip().split(sep_index)]
    ids_test = [int(id) for id in train_test[1].strip().split(sep_index)]    
    x_train = [datas[index].toarray()[0] for index in ids_train]
    x_test = [datas[index].toarray()[0] for index in ids_test]    
    return x_train, x_test

def remove_caracters_especiais_por_espaco(text):
    text =  re.sub("[()!;':?><,.?/+-=-_#$%ˆ&*]", " ", text)
    return re.sub(' +', ' ', text) # remove multiple space