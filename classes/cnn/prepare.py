

#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

onlyfiles = [f for f in listdir(wdir_data+'db/UMA_ADL_FALL_Dataset/') if isfile(join(wdir_data+'db/UMA_ADL_FALL_Dataset/', f))]
onlyfiles = np.sort(onlyfiles)[2:]

class_l = ['Bending', 
           'forwardFall',
           'Hopping',
           'Jogging',
           'LyingDown',
           'Sitting_GettingUpOnAChair',
           'Walking',
           'Fall_backwardFall',
           'Fall_lateralFall'#,
           #'GoDownstairs',
           #'GoUpstairs'
           ]

size_class = len(class_l)

class_l = [f.lower() for f in class_l]

labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(class_l)

'''
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"]) 
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
'''

X_aux = []
X = []
X_features = []

Y_aux = []
Y = []
DF = []

def label(e):
    for l in class_l:
        if l in e.lower() :
            return l
    return 'fail'

def get_sensor(E, type):
    for e in E:
        if 'Sensor ID' in e:
            #print(E.loc[E[e] == 0])
            return E.loc[E[e] == type]

            
def mad_std(x):
    MAD = robust.mad(x)
    p = 0.0    
    for e in x:
        p = p + np.abs(e-MAD)**2
    p = np.sqrt(p/(len(x) - 1))
    return p

for e in onlyfiles:
    dataset = pd.read_csv(wdir_data+'db/UMA_ADL_FALL_Dataset/'+e, sep=';', header=32)
    if label(e) != 'fail':
        DF.append(dataset)
        Y_aux.append(label(e))

print ('Iniciando codificacao dos dados : X = {} ; Y = {} '.format(len(DF), len(Y_aux)))

for i in tqdm(DF):
    aux = []
    for index, e in i.iterrows():
        if e[6] == 0:
            aux.append([e[2], e[3], e[4]])
    X_aux.append(np.array(aux))

#X_aux = np.array(X_features)

for index, e in enumerate(X_aux):
    #print (X_aux[index].shape)
    X_aux[index] = X_aux[index][:X_aux[index].shape[0]/size_batch*size_batch]
    
Y_aux = labelEncoder.transform(Y_aux)
Y_aux_2 = []

ii = []
jj = []
kk = []

for index, e in enumerate(X_aux):

    for i in e.reshape((e.shape[0]/size_batch,size_batch,3)):

       X.append(i)#(i-np.mean(i))/np.std(i))
       ii = [element[0] for element in i];ii = np.array(ii)
       jj = [element[1] for element in i];jj = np.array(jj)
       kk = [element[2] for element in i];kk = np.array(kk)
       #print ii

#       ii = (ii - np.mean(ii))/np.std(ii)
#       jj = (jj - np.mean(jj))/np.std(jj)
#       kk = (kk - np.mean(kk))/np.std(kk)
       #print ii

       array_aux = []
       for aux_index in range(len(ii)):
           array_aux.append([ii[aux_index], jj[aux_index], kk[aux_index]])
       array_aux = np.array(array_aux)
#       X.append(array_aux)
#       print len(ii), len(jj), len(kk)
#       print [np.mean(i), np.std(i), iqr(i), robust.mad(i)]

       X_features.append(np.array([np.mean(ii), np.mean(jj), np.mean(kk), np.std(ii), np.std(jj), np.std(kk),skew(i)[0], skew(i)[1], skew(i)[2], kurtosis(i)[0], kurtosis(i)[1], kurtosis(i)[2], iqr(ii), iqr(jj), iqr(kk), robust.mad(i)[0], robust.mad(i)[1], robust.mad(i)[2], mad_std(ii),mad_std(jj), mad_std(kk)]))

       aux = np.zeros(size_class)
       aux[Y_aux[index]] = 1
       Y_aux_2.append(Y_aux[index])
       Y.append(aux)



X = np.array(X)+1
Y = np.array(Y)

#--------------------------------------------------------

from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()
scaler.fit(X[0])

X[0] = scaler.transform(X[0])

scaler = StandardScaler()  
scaler.fit(X[1])

X[1] = scaler.transform(X[1])  

scaler = StandardScaler()
scaler.fit(X[2])

X[2] = scaler.transform(X[2])


#----------------------------------------------------------

Y_aux_2 = np.array(Y_aux_2)
X_features  = np.array(X_features)

X_features_train = []
X_features_test =  []



sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=1)

for train, test in sss.split(X, Y):
    X_train, X_test = X[train], X[test]
    y_train, y_test, y_aux_train, y_aux_test = Y[train], Y[test], Y_aux_2[train], Y_aux_2[test]
    X_features_train = X_features[train]
    X_features_test = X_features[test]

from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  
scaler.fit(X_features_train)

X_features_train = scaler.transform(X_features_train)  
X_features_test = scaler.transform(X_features_test) 


X_features_train = X_features_train.reshape(X_features_train.shape[0], X_features_train.shape[1], 1)
X_features_test = X_features_test.reshape(X_features_test.shape[0], X_features_test.shape[1], 1)



#X_train = X[ : train_size]
#y_train = Y[ : train_size]
#y_aux_train = Y_aux_2[ : train_size]

#X_test  = X[train_size : ]
#y_test  = Y[train_size : ]
#y_aux_test = Y_aux_2[train_size : ]

X_train = np.array(X_train)
y_train = np.array(y_train)
y_aux_train = np.array(y_aux_train)

X_test  = np.array(X_test )
y_test  = np.array(y_test )
y_aux_test = np.array(y_aux_test)

#X_features_train = np.array(X_features_train).reshape(X_features_train.shape[0], X_features_train.shape[1], 1).astype('float32')
#X_features_test = np.array(X_features_test).reshape(X_features_test.shape[0], X_features_test.shape[1], 1).astype('float32')

print 'shape = ', X_features_train.shape
print 'shape_test = ', X_features_test.shape

time.sleep(5)



#X_train = X_train.reshape(X_train.shape[0], 144,1).astype('float32')
#X_test  = X_test.reshape (X_test.shape[0], 144,1).astype('float32')

#X_train = X_train / float(descr_size)
#X_test  = X_test / float(descr_size)
