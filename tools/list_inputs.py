import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.compat.v1.Session() as sess:
    with gfile.FastGFile("AnimeGANv2/output/generator_Hayao.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        print("🔍 Nœuds d'entrée disponibles :")
        for op in sess.graph.get_operations():
            if op.type == "Placeholder":
                print(f"➡ {op.name}")
