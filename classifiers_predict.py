from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm  # Classifier SVN
from sklearn.naive_bayes import MultinomialNB  # Classifier MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier # Classifier ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb 
import scipy # load format sparse
from sklearn.datasets import load_svmlight_file, load_svmlight_files # import file svmlight
import numpy # Manipulate array
import timeit  # Measure time
import sys # Import other directory
import claudio_funcoes_sub as cv  # Functions utils author
import joblib
#from bert_sklearn import BertClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
import joblib

ini = timeit.default_timer() # Time process

""" Example execution: python3.6 svm_predict.py name_dataset file_train file_test num_folds"""

def predict_classifier(name_dataset, name_train, classifier, name_test, metric, index):
    """Run classifier"""            
    if classifier == "ada_boost":        
        estimator = AdaBoostClassifier(random_state=42, base_estimator= ComplementNB(alpha=0.01))
        #estimator = AdaBoostClassifier(random_state=42, base_estimator= LogisticRegression(C= 50, max_iter= 100))
    elif classifier == "xgboost": 
        estimator = xgb.XGBClassifier()   
    
    elif classifier == "bert": 
        estimator = BertClassifier(eval_batch_size=16, train_batch_size=16, epochs=3, learning_rate=3e-5)

    elif classifier == "extra_tree":        
        estimator = ExtraTreesClassifier(random_state=SEED)
    
    elif classifier == "knn":        
        estimator = KNeighborsClassifier()
    
    elif classifier == "logistic_regression":        
        estimator = LogisticRegression(random_state=SEED, max_iter=1000)
    
    elif classifier == "naive_bayes":        
        estimator = MultinomialNB()
    
    elif classifier == "naive_bayes_complement":        
        estimator = ComplementNB()
    
    elif classifier == "passive_aggressive":        
        estimator = PassiveAggressiveClassifier(random_state=SEED, max_iter=1000)
    
    elif classifier == 'decision_tree':
        estimator = DecisionTreeClassifier(random_state=SEED)#, class_weight={0:1, 1:20})


    elif classifier == "random_forest":        
        estimator = RandomForestClassifier(random_state=SEED)        
        
    elif classifier == "sgd":        
        estimator = SGDClassifier(random_state=SEED, max_iter=1000)

    elif classifier == "svm":        
        #estimator = svm.SVC(**cv.svm_init_lbd)
        #estimator = svm.SVC(random_state=SEED, max_iter=10000, probability=True)
        #estimator = svm.SVC(random_state=SEED, max_iter=1000, probability=True, kernel='linear') # scikit informa que nao e tao confiavel probability para svm, pode variar o predict e proba
        estimator = svm.LinearSVC(random_state=SEED, max_iter=1000)

    elif len(classifier.split(",")) > 1:        
        best = []
        classifiers = classifier.split(",")        
        for cl in classifiers:            
            escores = cv.load_escores(name_dataset, cl, 1) # test score 0
            best.append(cv.best_param_folds_no_frequency(escores, 0, metric)) # best score per fold 
        #stanford_svm_macro = {'C': 50, 'gamma': 0.01, 'kernel': 'rbf'}
        #stanford_passive_macro = {'C': 1, 'fit\_intercept': True}

        #stanford, reut, acm, 20ng
        #classifiers = [('svm', svm.SVC(**best[0])), ('passive_aggressive', PassiveAggressiveClassifier(**best[1])), ('naive_bayes_complement', ComplementNB(**best[2]))]
        #classifiers = [('sgd', SGDClassifier(**best[0])),  ('passive_aggressive', PassiveAggressiveClassifier(**best[1])), ('svm', svm.SVC(**best[2]))]
        #classifiers = [('svm', svm.SVC(**best[0])), ('sgd', SGDClassifier(**best[1])),  ('logistic_regression', LogisticRegression(**best[2]))]
        #classifiers = [('naive_bayes', MultinomialNB(**best[0])), ('sgd', SGDClassifier(**best[1])), ('svm', svm.SVC(**best[2]))] 

        best_hy = {'alpha': 1.017335258872783e-05}   
        #best[0].update({'C': 1})
        best[0].update({'random_state': SEED})
        best[1].update({'random_state': SEED})
        #best[2].update({'random_state': SEED})

        #classifiers = [('svm', svm.LinearSVC(**best[0])), ('passive_aggressive', PassiveAggressiveClassifier(**best[1])), ('sgd', SGDClassifier(**best[2])) ]

        #micro f1
        #classifiers = [('svm', svm.SVC(**best[0])), ('passive_aggressive', PassiveAggressiveClassifier(**best[1])), ('naive_bayes_complement', ComplementNB(**best[2]))]
        #classifiers = [('naive_bayes', MultinomialNB(**best[0])), ('naive_bayes_complement', ComplementNB(**best[1])), ('logistic_regression', LogisticRegression(**best[2]))  ]
        #classifiers = [('naive_bayes', MultinomialNB(**best[0])), ('naive_bayes_complement', ComplementNB(**best[1])),  ('sgd', SGDClassifier(**best[2]))  ]
        #classifiers = [('sgd', SGDClassifier(**best[0])), ('naive_bayes_complement', ComplementNB(**best[1])),  ('naive_bayes', MultinomialNB(**best[2]))  ]
        #classifiers = [('sgd', SGDClassifier(**best_hy)), ('passive_aggressive', PassiveAggressiveClassifier(**best[1])), ('naive_bayes_complement', ComplementNB(**best[2])) ]
        #classifiers = [('sgd', SGDClassifier(**best[0])),  ('logistic_regression', LogisticRegression(**best[1])),  ('naive_bayes', MultinomialNB(**best[2]))  ]
        #classifiers = [('logistic_regression', LogisticRegression(**best[0])),  ('naive_bayes', MultinomialNB(**best[1]))  ]
        #classifiers = [('naive_bayes', MultinomialNB(**best[0])), ('sgd', SGDClassifier(**best[1])), ('logistic_regression', LogisticRegression(**best[2]))  ]
        #classifiers = [('svm', svm.SVC(**best[0])), ('sgd', SGDClassifier(**best[1])), ('naive_bayes_complement', ComplementNB(**best[2]))]
        #classifiers = [('sgd', SGDClassifier(**best[0])),  ('logistic_regression', LogisticRegression(**best[1])), ('svm', svm.SVC(**best[2]))]
        #classifiers = [('passive_aggressive', PassiveAggressiveClassifier(**best[0])), ('sgd', SGDClassifier(**best[1])),  ('logistic_regression', LogisticRegression(**best[2])) ]
        #classifiers = [('passive_aggressive', PassiveAggressiveClassifier(**best[0])), ('sgd', SGDClassifier(**best[1])) ]

        classifiers = [('passive_aggressive', PassiveAggressiveClassifier(**best[0])), ('svm', svm.LinearSVC(**best[1])) ]
        #classifiers = [('naive_bayes', MultinomialNB(**best[0])), ('sgd', SGDClassifier(**best[1])), ('svm', svm.SVC(**best[2]))]
        #classifiers = [('naive_bayes', MultinomialNB(**best[0])), ('sgd', SGDClassifier(**best[1])), ('svm', svm.SVC(**best[2])), ('passive_aggressive', PassiveAggressiveClassifier(**best[3]))]
        #classifiers = [('naive_bayes', MultinomialNB(**best[0])), ('naive_bayes_complement', ComplementNB(**best[1])), ('passive_aggressive', PassiveAggressiveClassifier(**best[2]))  ]
        #classifiers = [('sgd', SGDClassifier(**best[0])),  ('passive_aggressive', PassiveAggressiveClassifier(**best[1])) ]

        #estimator = VotingClassifier(estimators=classifiers, voting="soft", n_jobs=-1)
        estimator = VotingClassifier(estimators=classifiers)#,   n_jobs=-1)
        #estimator = VotingClassifier(estimators=classifiers, weights=[1.0,0.75,0.50], n_jobs=-1)
    
    
    if classifier != 'bert':
        # saude ufmg comentar
        #if classifier == 'xgboost': 
        #    x_train = scipy.sparse.load_npz(name_train + '_xgb.npz'); x_test = scipy.sparse.load_npz(name_test + '_xgb.npz')
        #    no_use, y_train, no_use2, y_test = load_svmlight_files([open(name_train, 'rb'), open(name_test, 'rb')])
        #else:
        x_train, y_train, x_test, y_test = load_svmlight_files([open(name_train, 'rb'), open(name_test, 'rb')])
    else:
        x_train = cv.load_dict_list(name_train)
        y_train = cv.load_dict_list(name_train +"_label")
        x_test = cv.load_dict_list(name_test)
        y_test = cv.load_dict_list(name_test +"_label")

    #scaler = StandardScaler(with_mean=False).fit(x_train)  # Armaneza configuracao da normalizacao       
    #x_test = scaler.transform(x_test)
    load_estimator = False
    if load_estimator == True: 
        estimator = joblib.load("escores/grid_"+name_dataset+"_"+classifier) # load estimator
    else:
        if not(len(classifier.split(",")) > 1) and classifier != 'bert':
            
            #escores = cv.load_escores(name_dataset, classifier, index) # melhor parametro do grid para cada fold            
            escores = cv.load_escores(name_dataset, classifier, 0)  # pegar escore do fold 0 para todos os folds
            best_param_folds = cv.best_param_folds_no_frequency(escores, 0, metric) # best score per fold  
            #print(best_param_folds)
            #exit()
            #best_param_svm = {'kernel':'linear', 'C':1}      
            #best_param_sgd = {'alpha': 0.0001, 'eta0': 0.0001, 'learning_rate': 'optimal', 'loss': 'hinge', 'max_iter': 1000, 'penalty': 'l2'}
            #best_param_svm = {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
            #estimator.set_params(**best_param_svm)
            
            '''from hyperopt import hp
            from BayesianOptimization import BayesianOptimization
            hyperparameters_sgd = {    
                'alpha': hp.uniform('C', 0.00001, 1),
                "loss" : hp.choice('loss', ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]),
                'learning_rate':  hp.choice('learning_rate', ['optimal', 'constant', 'invscaling', 'adaptive']),
                "eta0":  hp.uniform('eta0', 0.00001, 1)
            }
            hyperparameters_nb = {
                "alpha": hp.loguniform('nb alpha', -3, 5)
            }
            bayer = BayesianOptimization(MultinomialNB(), x_train, y_train)
            best_param_folds = bayer.fit(hyperparameters_nb)'''
            
            #best_param_folds = {'alpha': 1.017335258872783e-05}            #sgd
            #best_param_folds = {'alpha': 2.788803935869088e-05, 'loss': 'modified_huber'}     
            #best_param_folds = {'C': 1}      
            #best_param_folds.update({'max_iter' : 1000})#, 'dual' : False})
            print(best_param_folds)
            estimator.set_params(**best_param_folds)

        #estimator = CalibratedClassifierCV(base_estimator=estimator, cv=3)
        #x_train = [x_train[index].toarray()[0] for index in range(x_train.shape[0])] # nao aceita feature negativa
        #x_test = [x_test[index].toarray()[0] for index in range(x_test.shape[0])] # nao aceita feature negativa
        #scaler = MinMaxScaler()
        #x_train = scaler.fit_transform(x_train)
        #x_test = scaler.transform(x_test)

        estimator.fit(x_train, y_train)
        #joblib.dump(estimator, f"save_model/{name_dataset}_{classifier}" ); exit() # save model
        dict_time.update({'time_train': (timeit.default_timer() - ini)})
        
    ini2 = timeit.default_timer()        

    y_pred = estimator.predict(x_test)     
    '''#proba
    y_prob = estimator.predict_proba(x_test)#; print( y_prob ); exit()
    cv.save_dict_list([y_test], [y_prob], 'y_prob/'+name_dataset+"_" +classifier +"_" +metric +"_" +cv.name_file(name_test))    
    y_pred = numpy.argmax(y_prob, axis=1)
    for index_y_pred in range(len( y_pred)): # coloca a predicao a classe real
        y_pred[index_y_pred] = estimator.classes_[y_pred[index_y_pred] ]    
    '''

    #y_pred = numpy.array( [ 1 if y_pred[index][1] > 0.3 else 0 for index in range( len( y_pred) )] )
    dict_time.update({"time_predict" : (timeit.default_timer() - ini2)})
    cv.save_dict_list([y_test], [y_pred], 'y_pred/'+name_dataset+"_" +classifier +"_" +metric +"_" +cv.name_file(name_test))
    
    

dict_time = dict()
SEED=42
numpy.random.seed(seed=SEED)
name_dataset = sys.argv[1]
name_train=sys.argv[2]
name_test=sys.argv[3]
classifier=sys.argv[4]
index = sys.argv[6]
predict_classifier(name_dataset, name_train, classifier, name_test, sys.argv[5], index)

cv.save_dict_file("y_pred/" +name_dataset +"_time_"+index , dict_time)
print("Time End: %f" % (timeit.default_timer() - ini))
