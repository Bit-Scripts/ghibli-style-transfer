import tensorflow as tf
from tensorflow.python.platform import gfile

pb_path = "AnimeGANv2/output/generator_Hayao.pb"

with tf.compat.v1.Session() as sess:
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    print("🔍 Derniers nœuds du graphe (candidats pour --outputs) :")
    for op in sess.graph.get_operations()[-30:]:  # Affiche les 30 derniers
        print("➡️", op.name, "| type:", op.type)
