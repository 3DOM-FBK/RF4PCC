import time
import pickle
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

feat_to_use = []     # Indices of the features to use. If n is the number of features, from 0 to n-1

def load_features(filepath):
    ''' Load the features indices from a .txt file
       
        Attributes:
            filepath (string)   :  Path to the .txt file
    '''
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                global feat_to_use
                feat_to_use = [int(t) for t in tokens]

def read_model(filepath):
    ''' Read the Random Forest model from a .pkl file

        Attributes:
            filepath (string)   :   Path to the .pkl file
    '''
    return pickle.load(open(filepath, 'rb'))

def read_data(filepath):
    ''' Read the point cloud to classify from a .txt file

        Attributes:
            filepath (string)   :   Path to the .txt file
        
        Return:
            X (np.array)   :    Point cloud and features
    '''
    X = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(' ')
            if 'nan' not in tokens:   
                X.append([float(t) for t_index, t in enumerate(tokens)])
    return np.asarray(X, dtype=np.float32)

def write_classification(X, Y, filename):
    ''' Write a classified point cloud

        Attributes:
            X (np.array)        :   Point cloud and features
            Y (np.array)        :   Classes
            filename (string)   :   Output file path
    '''
    with open('{}.txt'.format(filename), 'w') as out:
        X = X.tolist()
        Y_pred = Y.tolist()
        for index, x in enumerate(X):
            x_as_str = " ".join([str(i) for i in x])
            out.write('{} {}\n'.format(x_as_str, str(Y_pred[index])))

def main():
    parser = argparse.ArgumentParser(description='Classify a point cloud with a random forest model.')
    parser.add_argument('features_filepath', help='Path to the file containing the index of the features and the class index')
    parser.add_argument('model', help='Path to .pkl file containing the trained model.')
    parser.add_argument('point_cloud', help='Path to .txt file containing the point cloud to classify.')
    parser.add_argument('output_name', help='Name of the predicted test file')
    args = parser.parse_args()

    start = time.time() 
    print ('Loading data ...')
    load_features(args.features_filepath)                   # Load feature indices
    model = read_model(args.model)                          # Load trained model
    X = read_data(args.point_cloud)                         # Load data to classify
    
    print ('Classifying the dataset ...')
    
    Y_pred = model.predict(X[:, feat_to_use])               # Classify the data

    print ('Saving ...')
    write_classification(X, Y_pred, args.output_name)       # Output classification
    end = time.time()
    print('Data classified in: {}'.format(end - start))


if __name__== '__main__':
    main()