def multi_class_SMOTE(X, y, n, random_state, verbose=1):
    """Using imblearn.over_sampling.SMOTE, performs (n-1) iterations of SMOTE to facilitate creating balanced target classes when multiple classes are present.

    Parameters
    ----------
    X : array-like
        Matrix containing the feature data to be sampled
    y : array-like (1-d)
        Corresponding target labels for each sample in X
    n : int
        Number of unique classes/labels in y
    random_state : int
        Value to set as the random_state for SMOTE function reproducibility
    verbose : int (1 or 2)
        If 1, prints label counts only after final SMOTE iteration
        If 2, prints label counts at each SMOTE iteration (including initial)

    Returns
    ----------
    X_resampled : array-like
        Matrix containing the resampled feature data
    y_resampled : array-like (1-d)
        Corresponding target labels for X_resampled
    """

    from imblearn.over_sampling import SMOTE
    import pandas as pd

    # Initialize a SMOTE object
    smote = SMOTE(random_state=random_state)

    # Output if verbose = 2
    if verbose == 2:
        print(f'Label counts for Original y:\n{pd.Series(y).value_counts()}')

    # Perform SMOTE n-1 times to achieve balanced target classes
    for i in range(n - 1):
        X, y = smote.fit_sample(X, y)

        # Print value counts after each step if verbose == 2
        if verbose == 2:
            print(
                f'Label counts after SMOTE # {i+1}:\n{pd.Series(y).value_counts()}')

    # Print final value counts if verbose == 1
    if verbose == 1:
        print(
            f'Label counts after SMOTE # {n-1}:\n{pd.Series(y).value_counts()}')

    X_resampled = X
    y_resampled = y

    return X_resampled, y_resampled


def train_test_acc_auc(X_train, X_test, y_train, y_test, y_hat_train, y_hat_test, clf, multi_class=False):
    """Returns classification accuracy score and ROC AUC score for both train and test data after train-test-split.

    Parameters
    ----------
    X_train : array-like
        Matrix containing the training feature data    
    X_test : array-like
        Matrix containing the testing feature data
    y_train : array-like (1-d)
        Corresponding target labels for each sample in X_train
    y_test : array-like (1-d)
        Corresponding target labels for each sample in X_test
    y_hat_train : array-like (1-d)
        Model predictions for each sample in X_train
    y_hat_test : array-like (1-d)
        Model predictions for each sample in X_test
    clf : Sklearn-type classifier object
        Classifier used to generate model predictions
    multi_class : Bool
        If True, computes AUC for multi-class classification problem

    Returns
    ---------
    Accuracy score and ROC AUC score for both training and test data.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    test_acc = accuracy_score(y_test, y_hat_test)
    train_acc = accuracy_score(y_train, y_hat_train)

    if multi_class:
        y_score_train = clf.predict_proba(X_train)
        auc_train = roc_auc_score(
            y_train, y_score=y_score_train, multi_class='ovr')

        y_score_test = clf.predict_proba(X_test)
        auc_test = roc_auc_score(
            y_test, y_score=y_score_test, multi_class='ovr')

    else:
        y_score_train = clf.predict_proba(X_train)
        auc_train = roc_auc_score(y_train, y_score=y_score_train)

        y_score_test = clf.predict_proba(X_test)
        auc_test = roc_auc_score(y_test, y_score=y_score_test)

    print(f'Training Accuracy Score: {round(train_acc,2)}')
    print(f'Training AUC: {round(auc_train,2)}\n')
    print(f'Testing Accuracy Score: {round(test_acc,2)}')
    print(f'Testing AUC: {round(auc_test,2)}')