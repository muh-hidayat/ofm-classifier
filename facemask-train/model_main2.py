import tensorflow as tf
import os
import numpy as np
from models.frontend import FacemaskClassifierModel
from data_loader.loader import DataLoader
from configs.config_handler import Config
from trainers.train import Train
from argparse import ArgumentParser
from utils.eval_model_tools import plot_confusion_matrix, plot_roc_curve
import tensorflow.compat.v1.keras.backend as K 
from matplotlib import pyplot as plt

tf.compat.v1.disable_eager_execution()



def main():
    '''
    Training a classifier and export confusion matrix and ROC curve.
    '''

    argparse = ArgumentParser()
    argparse.add_argument('--config', type=str, help='json config file path', default='configs/config.json')
    args = argparse.parse_args()
    config_path = args.config
    print("-_- -_- -_- -_- -_- -_- -_- Running %s -_- -_- -_- -_- -_- -_- -_-" % config_path)
    cfg = Config(path=config_path)
    data_generator = DataLoader(cfg)

    print('_- -_- -_- -_- -_- -_- -_- -_- Training Phase -_- -_- -_- -_- -_- -_- -_- -_-')
    # Uncomment the below code if you want to train the models
    tf.compat.v1.disable_eager_execution()
    classifier = FacemaskClassifierModel(cfg)
    # classifier = ClassifierModel(cfg)
    model = classifier.model
    trainer = Train(model, data_generator, cfg)
    trainer.train()

    log_dir = os.path.join(cfg.SAVED_FOLDER, cfg.MODEL_NAME)
    
    print("The model is trained successfully, Confusion Matrix and ROC Curve will be exported at '{}'".format(log_dir))
    print("-_- -_- -_- -_- -_- -_- -_- Create Confusion Matrix -_- -_- -_- -_- -_- -_- -_-")
    print("Confution Matrix and ROC curve will be saved at: ", log_dir)
    scores = model.predict(data_generator['test'])
    y_pred = np.argmax(scores, axis=1)
    labels = data_generator['test'].classes
    plot_confusion_matrix(y_pred, labels, 2, log_dir)

    one_hot_lbl = np.eye(cfg.NO_CLASSES)[labels.reshape(-1)]

    print("-_- -_- -_- -_- -_- -_- -_- Create ROC Curve -_- -_- -_- -_- -_- -_- -_-")
    plot_roc_curve(one_hot_lbl, scores, cfg.NO_CLASSES, out_path=log_dir)

    # Plot history: MAE
    print(model.history.history['loss'])
    print(model.history.history['val_loss'])
    print(model.history.history['accuracy'])
    print(model.history.history['val_accuracy'])

    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

        # Plot history: MSE
    plt.plot(model.history.history['accuracy'], label='Training Accuracy')
    plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()
