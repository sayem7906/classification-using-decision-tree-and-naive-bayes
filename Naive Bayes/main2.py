from BayesianClassifier import NaiveBayes
import pandas as pd 	
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def accuracy_score(y_true, y_pred):

	"""	score = (y_true - y_pred) / len(y_true) """

	return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def pre_processing(df):

	""" partioning data into features and target """

	X = df.drop(columns=['class'], axis = 1)
	y = df['class']

	return X, y
def read_file_to_dict(filename):
    att_dict = {}
    f = open(filename, "r")
    i =0
    for line in f:
        words = line.split()
        att_dict[words[0]] = words[1]
        label = words[0]
    return att_dict, label

datasets = ['zoo.data','iris.data','wine.data','cmc.data','crx.data','german.data','krkopt.data','adult.data']
attributes = ['zoo_att.txt','iris_att.txt','wine_att.txt','cmc_att.txt','crx_att.txt','german_att.txt','krkopt_att.txt','adult_att.txt']

open('../Files/result_naive.csv','w').close()
ct = 0

for dataset_path,att_path in zip(datasets,attributes):
    att_full_path = "../datesets_classifier/"+att_path
    print(att_full_path)
    attribute,last = read_file_to_dict(att_full_path)
    dataset_full_path = '../datesets_classifier/'+dataset_path
    print(dataset_full_path)
    df = pd.read_csv(dataset_full_path,
                          names=list(attribute.keys()))
    
    #print(df)
       	#Split fearures and target
    X,y  = pre_processing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
    
    #print(X)
    #print(y)
    nb_clf = NaiveBayes()
    nb_clf.fit(X_train, y_train,attribute)
    result = nb_clf.predict(X_test,attribute)
    # print(result)
    # print("---------------------------\n")
    # print(y_test)
    # print("---------------------------\n")
    title = 'dataset,Accuracy %,Precision %,Recall %,f1 %\n'
    outf = open('../Files/result_naive.csv','a')
    if(ct==0):
        outf.write(title)
        ct+=1
    precision = 0
    recall = 0
    f1 = 0
    accuracy = accuracy_score(y_test,result)
    print("Test Accuracy: {}".format(accuracy))
    precision = precision_score(y_test, result, average='macro')*100
    print("Test Precision: {}".format(precision))
    recall = recall_score(y_test, result, average='macro')*100
    print("Test Recall: {}".format(recall))
    f1 = 2 * (precision * recall) / (precision + recall)
    print("Test F1 score: {}".format(f1))
    buffer_s = dataset_path.replace('.data','') + ',' + str(accuracy) + ',' + str(precision) +','+ str(recall) +','+ str(f1) + '\n'
    outf.write(buffer_s)
    outf.close()