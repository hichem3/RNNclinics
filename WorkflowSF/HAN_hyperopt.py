# script to run hyperopt on HAN (build and data from related scripts)
# Authors: Enrico Sartor, Loic Verlingue

#####################################################
# Model Training                                    #
#####################################################
i = 0

'''
    arguments HAN :
    self, max_words, max_sentences, output_size,
    embedding_matrix, word_encoding_dim,
    sentence_encoding_dim,
    l1=0., l2=0., dropout=0.,
    inputs=None,
    outputs=None, name='han-for-docla'

    '''

def create_model(params):
    global i
    i += 1
    logger.info('Model' + str(i) + 'training')

    l2 = params['l2']
    dropout = params['dropout']
    lr = params['lr']

    han_model = HAN(
        MAX_WORDS_PER_SENT, MAX_SENT, 1, embedding_matrix,  # 1 is output size
        # int(params['word_encoding_dim']),
        # int(params['word_encoding_dim']), #number of units for the 2 GRUs
        word_encoding_dim,
        sentence_encoding_dim,
        l1,  # float(params['regularization']['l1']),
        l2,  # float(params['regularization']['l2']),
        dropout  # float(params['regularization']['dropout'])
    )

    han_model.summary()

    opt = Adam(lr=lr)

    han_model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['acc', rec_scorer, f1_scorer, f2_scorer])

    # print(han_model.metrics_names)
    # es = EarlyStopping(monitor='val_loss', mode='min', patience = 5, verbose=1)
    # mc = ModelCheckpoint('best_HAN.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # up here 'best_HAN_'+str(i)+'.h5'

    han_model.fit(X_train, y_train, validation_split=0, batch_size=8,
                  epochs=Nepochs)  # ,callbacks=[es, mc])

    scores = han_model.evaluate(X_test, y_test, verbose=0)

    f2 = scores[4]
    f1 = scores[3]
    rec = scores[2]
    accuracy = scores[1]
    loss = scores[0]
    
    # build and save results and parameters
    df_scores=pd.DataFrame([scores],columns=('loss','accuracy','recall','f1','f2'))
    df_new=df_scores.join(pd.DataFrame.from_dict(params))
    
    try:
        df_results=df_results.append(df_new)
        df_results.to_csv(os.path.join(results_dir, out_file+'all.csv'))
    except NameError:
        df_results=df_new
        df_results.to_csv(os.path.join(results_dir, out_file+'all.csv'))
    
    # Save the best model
    if f1 <= df_results['f1'].min():
        # han_model.save("han_model.hd5")
        print("Save model")
        han_model.save(os.path.join(results_dir, "han_model.hd5"))

    return {'loss': loss, 'params': params, 'status': STATUS_OK}


space = {
    'l2': hp.qloguniform('l2', np.log(0.00001), np.log(0.01), 0.00001),
    'dropout': hp.quniform('dropout', 0, 0.5, 0.2),
    'lr': hp.qloguniform('lr', np.log(0.00001),  np.log(0.05), 0.00001)
}

# Trials object to track progress
bayes_trials = Trials()

# Optimize
best = fmin(fn=create_model, space=space, algo=tpe.suggest, max_evals=MAX_EVALS,
            trials=bayes_trials)

print(best)
