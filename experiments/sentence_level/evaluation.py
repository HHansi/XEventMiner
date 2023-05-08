from sklearn.metrics import f1_score, recall_score, precision_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def sentence_label_evaluation(predictions, real_values):
    return macro_f1(real_values, predictions), weighted_f1(real_values, predictions)


def print_evaluation(df, pred_column, real_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    labels = set(real_values)

    for label in labels:
        print()
        print("Stat of the {} Class".format(label))
        print("Recall {}".format(recall_score(real_values, predictions, labels=labels, pos_label=label)))
        print("Precision {}".format(precision_score(real_values, predictions, labels=labels, pos_label=label)))
        print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label=label)))

    print()
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighter F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))