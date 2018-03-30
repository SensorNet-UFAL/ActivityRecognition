#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import numpy

#max_review_length = 1600
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

#embedding_vecor_length = 300

print 'Iniciando compilação'

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data

nfolds=6
input_shape= (21, 1)

sum_score = 0

train_size = 2500


size_batch = 2960  

wdir_data ='../../' 

print ('Iniciando preparação dos dados .')

execfile('definition.py')

#print 'Iniciando ' + sys.argv[1]

def main():
    if(sys.argv[1] == 'train'):
         Train(input_shape)
    if(sys.argv[1] == 'predict'):
      print ('Predict ')
      o, m = Predict(X_features_test)
      for e in m:
                print e/np.sum(e)
         #plt.figure()
      plot_confusion_matrix(m, classes = class_l, title='Normalized confusion matrix', normalize = True)		
      plt.show()
if __name__ == '__main__':
    main()
    
