import tensorflow as tf
import keras.backend as k

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras.models import load_model


# Freeze graph process to change variables into constants to export to
# Android or iOS applications


def freeze_graph(session, keep_var_name=None, output_names=None, clear_devices=True):
    graph = session.graph
    # session, the way we actually run our graph itself
    # graph variable stores all the layers and the hidden nodes of the model

    with graph.as_default():
        # Bunch of variable names that we want to convert to constants

        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).
                                difference((keep_var_name or [])))
        # Getting all of the operation names for each of the variables in the tensorflow global variables
        # Conv2D, MaxPooling, Dense, Dropout Layers contains a lot of different operations
        # The operations are stored within variables nodes
        # So we are getting all the variables within our current graph using tf.global_variables
        # Getting each of those variables, v
        # Then getting the operation name, v.op.name
        # set, is created for those, to make sure there are no repeats
        # then making a list of the DIFFERENCE between the set(v.op.name for v in tf.global_variables()) and
        # the (keep_var_name or an empty list)
        # If keep_var_name, the argument, is None, then we start with an empty list
        # To keep it simple, freeze_var_names variable will just be a list of all the variables of
        # operation names
        # It freezes those trained values

        output_names = output_names or []
        # If output_names, argument is None, then we start with an empty list

        output_names += [v.op.name for v in tf.compat.v1.global_variables()]

        input_graph_def = graph.as_graph_def()
        # input_graph_def, The definition/shape of the graph, as well as all the trained values

        if clear_devices:
            # If clear_devices is True, we're just gonna clear the device for
            # each of the nodes in our inputs graph definition
            for node in input_graph_def.node:
                node.device = ''

        frozen_graph = convert_variables_to_constants(sess=session,
                                                      input_graph_def=input_graph_def,
                                                      output_names=output_names,
                                                      ariable_names_whitelist=freeze_var_names)

        return frozen_graph


model = load_model(filepath='Image_Classifier.h5')

# k.set_learning_phase(1)

graph_frozen = freeze_graph(k.get_session(), output_names=[model.output.op.name])
tf.io.write_graph(graph_frozen, '.', 'Image_Classifier.pb', False)
