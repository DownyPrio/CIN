import tensorflow as tf
import numpy as np

import CIN_model as CIN
import CIN_layer as Layer


trainSet=np.array([[[1,2,3]],
                   [[2,4,6]]]).astype("float64")
labelSet=np.array([[[0.1]],
                   [[0.2]]]).astype("float64")
x=tf.placeholder(shape=(1,3),dtype=tf.float64)
y=tf.placeholder(shape=(1,1),dtype=tf.float64)
x=tf.constant(np.array([[1,2,3]]).astype("float64"))

model=CIN.CIN_model([Layer.CIN_layer(3,input_shape=(1,3)),Layer.CIN_layer(2)],x)
y_pred=model.predict(x)
with tf.Session() as sess:
    print(sess.run(y_pred,feed_dict={x:trainSet}))
