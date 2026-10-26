import numpy as np
import h5py

import torch
import torch.nn.functional as F


class ClassificationHandler():
    def __init__(self, dataset_filename, criterion):
        # Task name
        self.name = 'classification'

        # Loss for this task
        self.criterion = criterion

        # Dict for saving results of each batch
        # (Will be { losses_wo_reduction:[...], preds:[...],
        # ys:[...], scores:[...], n_right:[...], samples:[...] })
        self.batches_results = dict()

        # Dict for saving results of each batch when we are
        # testing the model in a dataset independant from the
        # one used to train the model
        # This dict is different because we don't have the labels
        # of samples, so we only save the model predictions
        # (we can't compute loss and n_right)
        # Dict will be {scores:[...], preds:[...], samples=[...]}
        self.indep_test_batches_results = dict()

        # Dict for remembering results of the best epoch obtained so far
        # {mean_loss = float, mean_acc = float}
        self.best_epoch_results = dict()

        # Label names
        f = h5py.File(dataset_filename, 'r')
        self.class_label_names = np.array(f['class_label_names'], dtype=np.str_)
        f.close()


    def compute_loss(self, model_out, y_batch):
        return self.criterion(model_out, y_batch)


    def format_ybatch(self, y_batch):
        return y_batch


    # To get int 64
    def get_label(self, f, index):
        y = np.array(f['class_labels'][index], dtype=np.int64)

        return y


    def init_batches_results(self, dataset, dataloader):
        # Init the batch results using arrays filled with -1
        self.batches_results['losses_wo_reduction'] = np.ones(len(dataloader))*-1
        self.batches_results['preds'] = np.ones(len(dataset))*-1
        self.batches_results['scores'] = np.ones((len(dataset), len(self.class_label_names)))*-1
        self.batches_results['n_right'] = np.ones(len(dataloader))*-1
        self.batches_results['ys'] = np.ones(len(dataset))*-1
        self.batches_results['samples'] = np.ones(len(dataset))*-1
    
    
    def init_indep_test_batches_results(self, dataset, dataloader):
        # Init the batch results using arrays filled with -1
        self.batches_results['preds'] = np.ones(len(dataset))*-1
        self.batches_results['scores'] = np.ones((len(dataset), len(self.class_label_names)))*-1
        self.batches_results['samples'] = np.ones(len(dataset))*-1


    def update_batches_preds(self, model_out, y_batch, bstart, bend, batch):
        with torch.no_grad():
            # Scores : 1 value per class
            # (softmax = all values for a sample sum to 1)
            scores = F.softmax(model_out, dim=1).detach().cpu().numpy()
            self.batches_results['scores'][bstart:bend] = scores

            # Pred : class prediction for each sample
            preds = np.argmax(scores, axis=1)
            self.batches_results['preds'][bstart:bend] = preds

            # N_right : nb of samples with class correctly predicted
            n_right = ((y_batch.detach().cpu().numpy() - preds) == 0).sum()
            self.batches_results['n_right'][batch] = n_right
    

    def update_indep_test_batches_preds(self, model_out, bstart, bend, batch):
        with torch.no_grad():
            # Scores : 1 value per class
            # (softmax = all values for a sample sum to 1)
            scores = F.softmax(model_out, dim=1).detach().cpu().numpy()
            self.batches_results['scores'][bstart:bend] = scores

            # Pred : class prediction for each sample
            preds = np.argmax(scores, axis=1)
            self.batches_results['preds'][bstart:bend] = preds


    def print_baseline_results(self, baseline_results):
        # Nb of samples
        nb_samples = len(baseline_results['samples'])

        # Baseline loss
        loss = baseline_results['losses_wo_reduction'].sum()/nb_samples

        # Baseline accuracy
        acc = baseline_results['n_right'].sum()/float(nb_samples)*100

        print('Baseline loss: {} baseline acc: {}%'.format(loss, acc),
              flush=True)


    def print_epoch_results(self, train_results, valid_results):
        """
        Input
            train_results, valid resutls: a copy of
            self.batches_results = {losses_wo_reduction, preds, samples, ys}

        Output
            None, prints the mean train and valid epoch loss and accuray
        """

        # Nb of samples
        nb_samples_train = len(train_results['samples'])
        nb_samples_valid = len(valid_results['samples'])

        # Loss
        loss_train = train_results['losses_wo_reduction'].sum()/nb_samples_train
        loss_valid = valid_results['losses_wo_reduction'].sum()/nb_samples_valid

        # Accuracy
        acc_train = train_results['n_right'].sum()/float(nb_samples_train)*100
        acc_valid = valid_results['n_right'].sum()/float(nb_samples_valid)*100

        print('train loss: {} train acc: {}%'
              '\nvalid loss: {} valid acc: {}%'.format(
              loss_train, acc_train, loss_valid, acc_valid), flush=True)


    def print_resumed_best_results(self, results):
        print('best valid loss: {} best valid acc: {}%'.format(
               results['mean_loss'], results['mean_acc']))


    def print_test_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])

        loss = results['losses_wo_reduction'].sum()/nb_samples
        acc = results['n_right'].sum()/float(nb_samples)*100

        print('Test loss: {} test acc: {}%'.format(loss, acc))


    def init_best_epoch_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])

        # Best loss
        self.best_epoch_results['mean_loss'] = results['losses_wo_reduction'].sum()/nb_samples

        # Best accuracy
        self.best_epoch_results['mean_acc'] = results['n_right'].sum()/float(nb_samples)*100


    def resume_best_results(self, results):
        self.best_epoch_results['mean_loss'] = results['mean_loss']
        self.best_epoch_results['mean_acc'] = results['mean_acc']


    #This function is not valid when drop_last=True
    def update_best_results(self, results):
        has_improved = False

        # Nb samples
        nb_samples = len(results['samples'])

        # Actual loss
        loss = results['losses_wo_reduction'].sum()/nb_samples
        # Actual acc
        acc = results['n_right'].sum()/float(nb_samples)*100

        # Improvement if actual acc is greater than best acc
        if acc > self.best_epoch_results['mean_acc']:
            has_improved = True
            # Update best acc
            self.best_epoch_results['mean_acc'] = acc
            # Update best loss
            self.best_epoch_results['mean_loss'] = loss

        # Improvement if acc is same and actual loss is less than best loss
        if ((loss < self.best_epoch_results['mean_loss']) & (acc == self.best_epoch_results['mean_acc'])):
            has_improved = True
            # Update best loss
            self.best_epoch_results['mean_loss'] = loss

        return has_improved


    def save_predictions(self, results, file_fullpath):
        np.savez(file_fullpath,
                 samples=results['samples'],
                 labels=results['ys'],
                 preds=results['preds'],
                 scores=results['scores'],
                 class_label_names=self.class_label_names)



class RegressionHandler():
    def __init__(self, dataset_filename, criterion):
        # Task name
        self.name = 'regression'

        # Loss for this task
        self.criterion = criterion

        # Dict for saving results of each batch
        # (Will be { losses_wo_reduction:[...], preds:[...],
        # ys:[...], samples:[...] })
        self.batches_results = dict()

        # Dict for saving results of each batch when we are
        # testing the model in a dataset independant from the
        # one used to train the model
        # This dict is different because we don't have the labels
        # of samples, so we only save the model predictions
        # (we can't compute loss)
        # Dict will be {preds:[...], samples:[...]}
        self.indep_test_batches_results = dict()

        # Dict for remembering results of the best epoch obtained so far
        # {mean_loss = float}
        self.best_epoch_results = dict()


    def compute_loss(self, model_out, y_batch):
        return self.criterion(model_out, y_batch)


    # RESHAPE : ADDING BATCH DIM
    def format_ybatch(self, y_batch):
        return y_batch.unsqueeze(1)


    # To get float32
    def get_label(self, f, index):
        y = np.array(f['regression_labels'][index], dtype=np.float32)

        return y


    def init_batches_results(self, dataset, dataloader):
        # Init the batch results with empty arrays
        self.batches_results['losses_wo_reduction'] = np.ones(len(dataloader))*-1
        self.batches_results['preds'] = np.ones(len(dataset))*-1
        self.batches_results['ys'] = np.ones(len(dataset))*-1
        self.batches_results['samples'] = np.ones(len(dataset))*-1


    def init_indep_test_batches_results(self, dataset, dataloader):
        # Init the batch results with empty arrays
        self.batches_results['preds'] = np.ones(len(dataset))*-1
        self.batches_results['samples'] = np.ones(len(dataset))*-1


    def update_batches_preds(self, model_out, y_batch, bstart, bend, batch):
        # Model out is dim batch_size x 1 (each output is it's own tensor)
        # .squeeze() makes the dim batch_size (1 array of all outputs)
        self.batches_results['preds'][bstart:bend] = \
                model_out.detach().squeeze().cpu().numpy()
    

    def update_indep_test_batches_preds(self, model_out, bstart, bend, batch):
        # Same as update_batches_preds() function below (it's using only the model outputs)
        # The function below requires y_batch argument (to match how it's defined in
        # the ClassificationHandler that uses y_batch to compute accuracy)
        # Since the function in regressionHandler does not use the arg, we just pass
        # None as value for that arg (because we don't have lables (ys) of samples
        # in the indep_test data)
        self.update_batches_preds(model_out, None, bstart, bend, batch)

    def print_baseline_results(self, baseline_results):
        # Nb of samples
        nb_samples = len(baseline_results['samples'])

        # Baseline loss
        loss = baseline_results['losses_wo_reduction'].sum()/nb_samples

        print('Baseline loss: {}'.format(loss), flush=True)


    def print_epoch_results(self, train_results, valid_results):
        """
        Input
            train_results, valid resutls: a copy of
            self.batches_results = {losses_wo_reduction, preds, samples, ys}

        Output
            None, prints the mean train and valid epoch loss
        """

        # Nb of samples
        nb_samples_train = len(train_results['samples'])
        nb_samples_valid = len(valid_results['samples'])

        # Loss
        loss_train = train_results['losses_wo_reduction'].sum()/nb_samples_train
        loss_valid = valid_results['losses_wo_reduction'].sum()/nb_samples_valid

        print('train loss: {} valid loss: {}'.format(
              loss_train, loss_valid), flush=True)


    def print_resumed_best_results(self, results):
        print('best valid loss: {}'.format(results['mean_loss']))


    def print_test_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])

        loss = results['losses_wo_reduction'].sum()/nb_samples

        print('Test loss: {}'.format(loss))


    def init_best_epoch_results(self, results):
        # Nb samples
        nb_samples = len(results['samples'])
        # Best loss
        self.best_epoch_results['mean_loss'] = results['losses_wo_reduction'].sum()/nb_samples


    def resume_best_results(self, results):
        self.best_epoch_results['mean_loss'] = results['mean_loss']


    def update_best_results(self, results):
        has_improved = False

        # Nb samples
        nb_samples = len(results['samples'])
        # Actual loss
        loss = results['losses_wo_reduction'].sum()/nb_samples

        # Improvement if actual loss is less than best loss
        if loss < self.best_epoch_results['mean_loss']:
            has_improved = True
            # Update best loss
            self.best_epoch_results['mean_loss'] = loss

        return has_improved


    def save_predictions(self, results, file_fullpath):
        np.savez(file_fullpath,
                 samples=results['samples'],
                 labels=results['ys'],
                 preds=results['preds'])
