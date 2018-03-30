#################################################################
def KFold_Predict(x_test,nfolds=10,batch_size=128):
    #model = mnist_model(input_shape)
    model = generate_model_2(input_shape)
    yfull_test = []
    for num_fold in range(1,nfolds+1):
        weight_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
            
        p_test = model.predict(x_test, batch_size = batch_size, verbose=2)
        yfull_test.append(p_test)
        
    result = np.array(yfull_test[0])
    for i in range(1, nfolds):
	print 'Nfold = ' + str(i)
        result += np.array(yfull_test[i])
    result /= nfolds
    return result


#################################################################

def Predict(x_test):
	output = KFold_Predict(x_test)
	matrix = [[0.0 for x in range(size_class)] for y in range(size_class)] 
	for i in range(0, output.shape[0]):
         max = np.max(output[i])
  #       print 'max=', max
 #        print output[i]
         for index, e in enumerate(output[i]):
#             print e
             if e == max:
                 y_var = index
                 break
         for index, e in enumerate(y_test[i]):
             if e == 1:
                 x_var = index
                 break
         matrix[x_var][y_var] += 1

	return output, matrix     


def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.array(cm)
    cm = cm.astype('int')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(u'Rotulo Verdadeiro')
    plt.xlabel(u'Rotulo Predito')
    plt.savefig('fig.pdf')    

