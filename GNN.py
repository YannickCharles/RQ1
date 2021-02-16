import tensorflow as tf
from graph_nets import _base
from graph_nets import blocks
import sonnet as snt


# TRANSITION GNN
class GraphNeuralNetwork_transition(_base.AbstractModule):
    """GNN-based transition function."""

    def __init__(self,
                 edge_model_fn,
                 node_model_fn,
                 name="fwd_gnn"):
        # aggregator_fn = tf.math.unsorted_segment_sum,
        """Initializes the transition model"""

        super(GraphNeuralNetwork_transition, self).__init__(name=name)

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
                use_sent_edges=True,
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



# TRANSITION GNN
class GraphNeuralNetwork_transition2(_base.AbstractModule):
    """GNN-based transition function."""

    def __init__(self,
                 edge_model_fn,
                 node_model_fn,
                 name="fwd_gnn"):
        # aggregator_fn = tf.math.unsorted_segment_sum,
        """Initializes the transition model"""

        super(GraphNeuralNetwork_transition, self).__init__(name=name)

        with self._enter_variable_scope():
            self._edge_block = blocks.EdgeBlock(
                edge_model_fn=edge_model_fn,
                use_edges=False,
                use_receiver_nodes=True,
                use_sender_nodes=True,
                use_globals=False,
                name='edge_block')

            # self._node_block = blocks.NodeBlock(
            #     node_model_fn=node_model_fn,
            #     use_received_edges=True,
            #     use_sent_edges=True,
            #     use_nodes=True,
            #     use_globals=True,
            #     received_edges_reducer=tf.math.unsorted_segment_sum,
            #     sent_edges_reducer=tf.math.unsorted_segment_sum,
            #     name="node_block")

            self._sent_edges_aggregator = blocks.SentEdgesToNodesAggregator(reducer=tf.math.unsorted_segment_sum)
            self._node_model = node_model_fn()


    def _build(self, graph):
        # representation of the node interaction:
        e_intermediate = self._edge_block(graph)

        # make f_node input:
        nodes_to_colllect = []
        nodes_to_colllect.append(graph.nodes)
        zero_action_nodes = tf.zeros(shape=(graph.n_node[0] - 1, 4), dtype=tf.dtypes.float32)
        action_on_nodes = tf.concat([zero_action_nodes, graph.globals], axis=0)
        nodes_to_colllect.append(action_on_nodes)
        # nodes_to_colllect.append(tf.repeat(graph.globals, graph.n_node[0], axis=0))
        nodes_to_colllect.append(self._sent_edges_aggregator(e_intermediate))

        # perform f_node:
        input_f_node = tf.concat(nodes_to_colllect, axis=-1)
        updated_nodes = self._node_model(input_f_node)

        graph_with_next_state = graph.replace(globals = None, nodes = updated_nodes) # remove globals form graph because only interested in the information on the nodes (i.e. state)
        return graph_with_next_state



# Reward GNN
class GraphNeuralNetwork_reward(_base.AbstractModule):
    """GNN-based reward function."""

    def __init__(self,
                 node_model_fn,
                 global_model_fn,
                 name="rwrd_gnn"):
        # aggregator_fn = tf.math.unsorted_segment_sum,
        """Initializes the reward model"""

        super(GraphNeuralNetwork_reward, self).__init__(name=name)

        with self._enter_variable_scope():
            # self._edge_block = blocks.EdgeBlock(
            #     edge_model_fn=edge_model_fn,
            #     use_edges=False,
            #     use_receiver_nodes=True,
            #     use_sender_nodes=True,
            #     use_globals=True,
            #     name='edge_block')

            self._node_block = blocks.NodeBlock(
                node_model_fn=node_model_fn,
                use_received_edges=False,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=True,
                received_edges_reducer=tf.math.unsorted_segment_sum,
                sent_edges_reducer=tf.math.unsorted_segment_sum,
                name="node_block")

            self._global_block = blocks.GlobalBlock(
                global_model_fn = global_model_fn,
                use_edges = False,
                use_nodes = True,
                use_globals = True,
                nodes_reducer = tf.math.unsorted_segment_sum,
                edges_reducer = tf.math.unsorted_segment_sum,
                name = "global_block")

    def _build(self, graph):
        """ Connects the network. """
        next_state_graph_representation = self._global_block(self._node_block(graph)) # feed graph through network
        reward = next_state_graph_representation.globals # use the global attribute to as a reward
        return reward