from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score


def display_metrics(y_pred, y):
    print("Displaying other metrics:")
    print("\t\tAccuracy (%)\tPrecision (%)\tRecall (%)\tF-measure (%)")
    print(
        f"Train:\t{round(accuracy_score(y, y_pred, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(y, y_pred, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(y, y_pred, average='macro') * 100, 2)}\t\t\t"
        f"{round(f1_score(y, y_pred, average='macro') * 100, 2)}")

