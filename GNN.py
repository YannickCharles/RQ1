import tensorflow as tf
from graph_nets import _base
from graph_nets import blocks
import sonnet as snt


# TRANSITION GNN
class GraphNeuralNetwork(_base.AbstractModule):
    """GNN-based transition function."""

    def __init__(self,
                 edge_model_fn,
                 node_model_fn,
                 name="fwd_gnn"):
        # aggregator_fn = tf.math.unsorted_segment_sum,
        """Initializes the transition model"""

        super(GraphNeuralNetwork, self).__init__(name=name)

        with self._enter_variable_scope():
            self._edge_block = blocks.EdgeBlock(
                edge_model_fn=edge_model_fn,
                use_edges=False,
                use_receiver_nodes=True,
                use_sender_nodes=True,
                use_globals=True,
                name='edge_block')

            self._node_block = blocks.NodeBlock(
                node_model_fn=node_model_fn,
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=True,
                received_edges_reducer=tf.math.unsorted_segment_sum,
                sent_edges_reducer=tf.math.unsorted_segment_sum,
                name="node_block")

    def _build(self, graph):
        """ Connects the network. """
        next_state_graph_representation = self._node_block(self._edge_block(graph)) # feed graph through network
        graph_with_next_state = next_state_graph_representation.replace(globals = None) # remove globals form graph because only interested in the information on the nodes (i.e. state)
        return graph_with_next_state