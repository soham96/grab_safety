from utils.helpers import read_test_data
from preprocess import prepare_data_for_training, get_test_batch
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve
import numpy as np
import argparse
from keras.models import load_model


def get_results(test_y, predicted_class):
    '''
    This is used to visualise the results of the predictions given by the model.
    Args:
        test_y: The actual label for that data point
        predicted_class: The variable predicted by the model
    Returns:
        NA. Only prints the statistics of the results.
    '''
    print('Confusion matrix is \n', confusion_matrix(test_y, predicted_class))
    print('tn, fp, fn, tp =')
    print(confusion_matrix(test_y, predicted_class).ravel())
    # Precision
    print('Precision = ', precision_score(test_y, predicted_class))
    # Recall
    print('Recall = ', recall_score(test_y, predicted_class))
    # f1_score
    print('f1_score = ', f1_score(test_y, predicted_class))
    # cohen_kappa_score
    print('cohen_kappa_score = ', cohen_kappa_score(test_y, predicted_class))
    # roc_auc_score
    print('roc_auc_score = ', roc_auc_score(test_y, predicted_class))

def test(args):
    model=load_model(args.model_path)

    x, y = read_test_data(args.data_folder, args.label_folder)

    counter=0
    predictions=[]
    test_y_class=[]

    print("Running Predictions. Depending on the amount of data, it might take some time.")

    pbar=tqdm(total=int(len(x['bookingID'].drop_duplicates())))
    while counter<=int(len(x['bookingID'].drop_duplicates())):
        test_x, test_y = next(prepare_data_for_training(x, y, seq_len=1200, model_type='LSTM', batch_size=1))
        preds=model.predict(test_x)

        predictions.extend([np.argmax(pred) for pred in preds])
        test_y_class.extend([np.argmax(r) for r in test_y])

        counter=counter+len(test_x)
        pbar.update(len(test_x))
    
    get_results(test_y_class, predictions)

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--model_path', help='Path to the model')
    a.add_argument('--data_folder', help='Folder where data is')
    a.add_argument('--label_folder', help='Folder where the label is')
    
    args = a.parse_args()
    test(args)

