import tensorflow as tf

# Tensorflow has an implicit default graph that we cannot
# access (_default_graph_stack) instead we use what is
# available.
graph = tf.get_default_graph()

# NODES of the TensorFlow grap are called "operations" or
# "ops" or "op".
# We can see what operations are in the graph with.
graph.get_operations()

# Currently nothing is in the graph so we have to add some
# value so that TensorFlow can compute into that graph.
# Start with a constant value
input_value = tf.constant(1.0)

# The constant is now a node or operation in the graph
# "input_value" is the op (operation)

# we can find the operation in the default graph
operations = graph.get_operations()
operations

# Printings the "node_def_ for the constant "input_value"
# show what in TensorFlow's protbuff representation for the
# float 1.0 that we assigned.

# uncomment code below to see output
# print (operations[0].node_def)

# If you print our costant "input_value", you
# will see that it is a cosntant 32-bit float of no
# dimension.

# uncomment code below to see output
# print (input_value)

# To see what the actual value of our constant we need
# to create a "session" where graph operations can be
# evaluated and then explicitly ask to run the constant
# "input_value".
# launch the graph in a session.
sess = tf.Session()
# Evaluate the tensor "inpute_value.
print (sess.run(input_value))
# It is important to release resoures when they are no longer required.
sess.close()
