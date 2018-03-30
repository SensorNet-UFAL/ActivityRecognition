#################################################################

def Train(input_shape = (72, 1)):
	
    x_train = X_train
    x_test = X_train
    x_t = []
    y_t = []


############################################################

#-------------------------DA--------------------------------

############################################################

	#a =np.empty((x_train.shape[0],28, 28, 3))
	#a[:][:][:][:]=x_train[:][:][:][:][:]
	#x_train = a

    KFold_Train(x_train,y_train)


#################################################################

def KFold_Train(x_train,y_train,nfolds=10,batch_size=128):
	model = generate_model_2(input_shape)
	print 'Modelo :'
    
	model.summary()
	time.sleep(5)    
	kf = KFold(n_splits=nfolds, shuffle=True, random_state=1)
	
	num_fold = 0 
	for train_index, test_index in kf.split(x_train, y_aux_train):

		model = generate_model_2(input_shape)

		start_time_model_fitting = time.time()
		X_train = x_train[train_index]
		Y_train = y_train[train_index]
		X_valid = x_train[test_index]
		Y_valid = y_train[test_index]
		X_f_train = X_features_train[train_index]
		X_f_test = X_features_train[test_index]
		
		#X_train = x_test.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
		
		#X_valid = x_test.reshape(X_valid.shape[0], 1, 28, 28).astype('float32')
		
		num_fold += 1
		print('Start KFold number {} from {}'.format(num_fold, nfolds))
		print('Split train: ', X_train.shape, X_f_train.shape, Y_train.shape)
		print('Split valid: ', X_valid.shape , X_f_test.shape, Y_valid.shape)
		
		kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

		epochs_arr =  [500, 300, 100, 20]
		learn_rates = [0.001, 0.0001, 0.00001, 0.000001]

		for learn_rate, epochs in zip(learn_rates, epochs_arr):
		    print('Start Learn_rate number {} from {}'.format(epochs,learn_rate))
		    opt  = optimizers.Adam(lr=learn_rate)
		    model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
		                  optimizer=opt,
		                  metrics=['accuracy'])
		    callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=2),
		    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

		    model.fit( X_f_train, Y_train, validation_data = (X_f_test, Y_valid),
		          batch_size=128,verbose=2, epochs=epochs,callbacks=callbacks,shuffle=True)
		
		if os.path.isfile(kfold_weights_path):
		    model.load_weights(kfold_weights_path)
            
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))


#################################################################


