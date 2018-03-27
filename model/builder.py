from __future__ import division
import os
import numpy as np
import tensorflow as tf


estimator_dir = os.path.join(os.path.dirname(__file__), '_model')


def _tuple_generator(nested_vals):
    iters = tuple(iter(nested_generator(v)) for v in nested_vals)
    try:
        while True:
            yield tuple(next(i) for i in iters)
    except StopIteration:
        pass


def _list_generator(nested_vals):
    iters = tuple(iter(nested_generator(v)) for v in nested_vals)
    try:
        while True:
            yield [next(i) for i in iters]
    except StopIteration:
        pass


def _dict_generator(nested_vals):
    iters = {k: iter(nested_generator(v)) for k, v in nested_vals.items()}
    try:
        while True:
            yield {k: next(i) for k, i in iters.items()}
    except StopIteration:
        pass


def nested_generator(nested_vals):
    if isinstance(nested_vals, np.ndarray):
        return nested_vals
    elif isinstance(nested_vals, (list, tuple)):
        if all(isinstance(v, str) for v in nested_vals):
            return nested_vals
        elif isinstance(nested_vals, tuple):
            return _tuple_generator(nested_vals)
        else:
            return _list_generator(nested_vals)
    elif isinstance(nested_vals, dict):
        return _dict_generator(nested_vals)
    else:
        raise TypeError(
            'Unrecognized type for nested_generator: %s'
            % str(type(nested_vals)))


class ModelBuilder(object):
    """
    Abstract base class for building models.

    Basically an umbrella class containing required functions to build data
    pipelines and `tf.estimator.Estimator`s.

    Concrete implementations must implement:
        * estimator construction:
            * get_inference
            * get_inference_loss
            * get_train_op
        * data pipelines:
            * get_train_dataset
            * get_eval_dataset
            * get_predict_dataset

    Implementations are encouraged to implement:
        * get_predictions
        * get_eval_metrics
        * vis_input_data
        * vis_prediction_data
    """

    def __init__(self, model_id, params):
        self._model_id = model_id
        self._params = params

    @property
    def model_id(self):
        return self._model_id

    @property
    def params(self):
        return self._params

    @property
    def model_dir(self):
        return os.path.join(estimator_dir, self.model_id)

    @property
    def batch_size(self):
        return self.params['batch_size']

    def get_inference(self, features, mode):
        """Get inferred value of the model."""
        raise NotImplementedError('Abstract method')

    def get_inference_loss(self, inference, labels):
        """Get the loss assocaited with inferences."""
        raise NotImplementedError('Abstract method')

    def get_train_op(self, loss, step):
        """
        Get the train operation.

        This operation is called within a `tf.control_dependencies(update_ops)`
        block, so implementations do not have to worry about update ops that
        are defined in the calculation of the loss, e.g batch_normalization
        update ops.
        """
        raise NotImplementedError('Abstract method')

    def vis_example_data(self, feature_data, label_data):
        """
        Function for visualizing a batch of data for training or evaluation.

        All inputs are numpy arrays, or nested dicts/lists of numpy arrays.

        Not necessary for training/evaluation/infering, but handy for
        debugging.
        """
        raise NotImplementedError()

    def vis_prediction_data(self, prediction_data, feature_data, label_data):
        """
        Function for visualizing a batch of data for training or evaluation.

        All inputs are numpy arrays, or nested dicts/lists of numpy arrays.

        `label_data` may be `None`.

        Not necessary for training/evaluation/infering, but handy for
        debugging.
        """
        raise NotImplementedError()

    def get_predictions(self, inferences):
        """Get predictions. Defaults to the identity, returning inferences."""
        return inferences

    def get_eval_metric_ops(self, predictions, labels):
        """Get evaluation metrics. Defaults to empty dictionary."""
        return dict()

    def get_total_loss(self, inference_loss):
        """
        Get total loss, combining inference loss and regularization losses.

        If no regularization losses, just returns the inference loss.
        """
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_losses) > 0:
            tf.summary.scalar(
                'inference_loss', inference_loss, family='sublosses')
            reg_loss = tf.add_n(reg_losses)
            tf.summary.scalar('reg_loss', reg_loss, family='sublosses')
            loss = inference_loss + reg_loss
        else:
            loss = inference_loss
        return loss

    def get_estimator_spec(self, features, labels, mode, config=None):
        """See `tf.estmator.EstimatorSpec`."""
        inference = self.get_inference(features, mode)
        predictions = self.get_predictions(inference)
        spec_kwargs = dict(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(**spec_kwargs)

        inference_loss = self.get_inference_loss(inference, labels)
        loss = self.get_total_loss(inference_loss)
        spec_kwargs['loss'] = loss

        if mode == tf.estimator.ModeKeys.EVAL:
            spec_kwargs['eval_metric_ops'] = self.get_eval_metric_ops(
                predictions, labels)
            return tf.estimator.EstimatorSpec(**spec_kwargs)

        step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = self.get_train_op(loss=loss, step=step)
        spec_kwargs['train_op'] = train_op

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(**spec_kwargs)

        raise ValueError('Unrecognized mode %s' % mode)

    def get_train_inputs(self):
        """
        Get all features and labels for training.

        Returns (features, labels), where each of (features, labels) can be
        a tensor, or possibly nested list/tuple/dict.
        """
        return self.get_inputs(mode=tf.estimator.ModeKeys.TRAIN)

    def get_eval_inputs(self):
        """
        Get all features and labels for evlauation.

        Returns (features, labels), where each of (features, labels) can be
        a tensor, or possibly nested list/tuple/dict.
        """
        return self.get_inputs(mode=tf.estimator.ModeKeys.EVAL)

    def get_predict_inputs(self):
        """
        Abstract method that returns all features required by the model.

        Returned value can be a single tensor, or possibly nested
        list/tuple/dict.
        """
        return self.get_inputs(mode=tf.estimator.ModeKeys.PREDICT)

    def get_inputs(self, mode):
        """
        Convenience function for calling inputs with different modes.

        Redirects calls to one of
            * `get_train_inputs`
            * `get_eval_inputs`
            * `get_predict_inputs`.
        """
        raise NotImplementedError()

    def get_estimator(self, config=None):
        """Get the `tf.estimator.Estimator` defined by this builder."""
        return tf.estimator.Estimator(
            self.get_estimator_spec, self.model_dir, config=config)

    def train(self, config=None, **train_kwargs):
        """Wrapper around `tf.estimator.Estimator.train`."""
        estimator = self.get_estimator(config=config)
        estimator.train(self.get_train_inputs, **train_kwargs)

    def predict(self, config=None, **predict_kwargs):
        """Wrapper around `tf.estimator.Estimator.predict`."""
        estimator = self.get_estimator(config=config)
        return estimator.predict(self.get_predict_inputs, **predict_kwargs)

    def eval(self, config=None, **eval_kwargs):
        """Wrapper around `tf.estimator.Estimator.eval`."""
        estimator = self.get_estimator(config=config)
        return estimator.evaluate(self.get_eval_inputs, **eval_kwargs)

    def vis_inputs(self, mode=tf.estimator.ModeKeys.TRAIN):
        """
        Visualize inputs defined by this model according.

        Depends on `vis_example_data` implementation.
        """
        graph = tf.Graph()
        with graph.as_default():
            if mode == tf.estimator.ModeKeys.PREDICT:
                features, labels = self.get_predict_inputs()
            elif mode == tf.estimator.ModeKeys.TRAIN:
                features, labels = self.get_train_inputs()

            with tf.train.MonitoredSession() as sess:
                while not sess.should_stop():
                    data = sess.run([features, labels])
                    for record in nested_generator(data):
                        self.vis_example_data(*record)

    def vis_predictions(self, mode=tf.estimator.ModeKeys.PREDICT):
        """
        Visualize inputs and predictions defined by this model.

        Depends on `vis_prediction_data` implementation.
        """
        graph = tf.Graph()
        with graph.as_default():
            if mode == tf.estimator.ModeKeys.PREDICT:
                features, labels = self.get_predict_inputs()
            elif mode == tf.estimator.ModeKeys.TRAIN:
                features, labels = self.get_train_inputs()

            predictions = self.get_estimator_spec(
                features, labels, tf.estimator.ModeKeys.PREDICT).predictions

            data_tensors = [predictions, features]
            if labels is not None:
                data_tensors.append(labels)
            saver = tf.train.Saver()

            with tf.train.MonitoredSession() as sess:
                saver.restore(
                    sess, tf.train.latest_checkpoint(self.model_dir))
                while not sess.should_stop():
                    data = sess.run(data_tensors)
                    for record in nested_generator(data):
                        self.vis_prediction_data(*record)
