def Print_model_metrics(exported_pipeline, X_train, X_test, cm_train, cm_test, auc_train, auc_test):
    true_negative_train  = cm_train[0, 0]
    true_positive_train  = cm_train[1, 1]
    false_negative_train = cm_train[1, 0]
    false_positive_train = cm_train[0, 1]
    total_train = true_negative_train + true_positive_train + false_negative_train + false_positive_train
    accuracy_train = (true_positive_train + true_negative_train)/total_train
    precision_train = (true_positive_train)/(true_positive_train + false_positive_train)
    recall_train = (true_positive_train)/(true_positive_train + false_negative_train)
    misclassification_rate_train = (false_positive_train + false_negative_train)/total_train
    F1_train = (2*true_positive_train)/(2*true_positive_train + false_positive_train + false_negative_train)
    
    true_negative_test  = cm_test[0, 0]
    true_positive_test  = cm_test[1, 1]
    false_negative_test = cm_test[1, 0]
    false_positive_test = cm_test[0, 1]
    total_test = true_negative_test + true_positive_test + false_negative_test + false_positive_test
    accuracy_test = (true_positive_test + true_negative_test)/total_test
    precision_test = (true_positive_test)/(true_positive_test + false_positive_test)
    recall_test = (true_positive_test)/(true_positive_test + false_negative_test)
    misclassification_rate_test = (false_positive_test + false_negative_test)/total_test
    F1_test = (2*true_positive_test)/(2*true_positive_test + false_positive_test + false_negative_test)
    
    y_predict_train = exported_pipeline.predict(X_train)
    y_predict_test  = exported_pipeline.predict(X_test)
    mse_train       = metrics.mean_squared_error(y_train, y_predict_train)
    mse_test        = metrics.mean_squared_error(y_test, y_predict_test)
    logloss_train   = metrics.log_loss(y_train, y_predict_train)
    logloss_test    = metrics.log_loss(y_test, y_predict_test)
    accuracy_train  = metrics.accuracy_score(y_train, y_predict_train)
    accuracy_test   = metrics.accuracy_score(y_test, y_predict_test)
    precision_test  = precision_score(y_test, y_predict_test, average='binary')
    recall_test     = recall_score(y_test, y_predict_test, average='binary')
    F1_train        = metrics.f1_score(y_train, y_predict_train)
    F1_test         = metrics.f1_score(y_test, y_predict_test)
    r2_train        = metrics.r2_score(y_train, y_predict_train)
    r2_test         = metrics.r2_score(y_test, y_predict_test)
    auc_train       = metrics.roc_auc_score(y_train, y_predict_train)
    auc_test        = metrics.roc_auc_score(y_test, y_predict_test)
    
    header = ["Metric", "Train", "Test"]
    table = [["accuracy",               accuracy_train,               accuracy_test],
             ["precision",              precision_train,              precision_test],
             ["recall",                 recall_train,                 recall_test],
             ["misclassification rate", misclassification_rate_train, misclassification_rate_test],
             ["F1",                     F1_train,                     F1_test],
             ["r2",                     r2_train,                     r2_test],
             ["AUC",                    auc_train,                    auc_test],
             ["mse",                    mse_train,                    mse_test],
             ["logloss",                logloss_train,                logloss_test]
            ]

    print(tabulate(table, header, tablefmt="fancy_grid"))