import sys # Import other directory
import claudio_funcoes_sub as cv  # Functions utils author

""" Example execution: python3.6 name_dataset num_folds"""

name_dataset = sys.argv[1]
folds=int(sys.argv[2])
y_pred_folds=[]
y_test_folds=[]
classifier=sys.argv[3]
metric = sys.argv[4]
best_param_folds = []

if len(classifier.split(",")) == 1 and classifier != 'bert':
    #pass # descomentar para bert
    #escores = cv.load_escores(name_dataset, classifier, 1) #escore fold 0, voting comentar
    for index_fold in range(folds): 
        break # stacking
        escores = cv.load_escores(name_dataset, classifier, 0)  #, 0   para pegar sempre o param do fold 0
        best_param_folds.append( cv.best_param_folds_no_frequency(escores, 0, metric) ) # voting comentar
else:
    best_param_folds = [] #voting

#print(best_param_folds)
#latex = name_dataset + "_" +classifier +"_" +metric +" = " +str(best_param_folds) + " \\\\"
#print(latex.replace("_", "\_"))
#exit()


time_method = []
for index in range(folds):
    fold = cv.load_dict_file('y_pred/'+name_dataset+"_" +classifier +"_" +metric +"_test"+str(index))    
    #fold = cv.load_dict_file('y_pred/'+name_dataset +"_" +classifier +"_test"+str(index) )  # stacking
    y_pred_folds.append(fold['y_pred-folds']['0'])
    y_test_folds.append(fold['y_true-folds']['0'])
    '''nohup = 'y_pred/'+name_dataset+"_nohup_" +classifier +"_" +metric +"_predict_test"+str(index)+".txt"
    with open(nohup, 'r') as read:
        for time in read:
            if time.__contains__("Time") == True:
                break
            pass                                    
        time = float(time[10:-1])        
        time_method.append(time)'''
time_method= [0]
cv.statistics_experiment(name_dataset, classifier, y_pred_folds, y_test_folds, best_param_folds, time_method, metric, folds)   

