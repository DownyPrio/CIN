import tensorflow as tf
import numpy as np

class CIN_model:
    def __init__(self,LayerList,input):
        self.layerList=LayerList
        self.input=input
        self.m0=self.layerList[0].input_shape[0]
        self.d=self.layerList[0].input_shape[1]
        self.listParse()
        self.build_w()


    def __call__(self, *args, **kwargs):
        return
    def build_w(self):
        self.w=[]
        for index in range(len(self.W_row_list)):
            print(self.W_row_list[index])
            print(self.H_list[index])
            tmp_w=tf.Variable(tf.zeros((self.W_row_list[index],self.H_list[index]),dtype=tf.float64))
            self.w.append(tmp_w)
    def listParse(self):
        self.H_list=[]
        self.W_row_list=[]
        for index in range(len(self.layerList)):
            self.H_list.append(self.layerList[index].H)
            if index==0:
                self.W_row_list.append(self.m0*self.m0)
            else:
                self.W_row_list.append(self.W_row_list[-1]*self.m0)
    def filer(self,Xn,X0):
        for index in range(self.d):
            print("index:" )
            print(index)
            mat_1=tf.convert_to_tensor([Xn[:,index]])
            mat_2=[X0[:,index]]
            mat_2_T=tf.transpose(mat_2)
            print(type(mat_1))
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     print(sess.run(mat_1))
            #     print(sess.run(mat_2_T))
            mat_d=tf.matmul(mat_1,mat_2_T)
            if index==0:
                mat=mat_d
            else:
                mat=tf.concat([mat,mat_d],axis=0)
        return mat

    def vec(self,mat):
        return tf.reduce_sum(mat,axis=0)
    def forward(self,tmp_mat,index):
        X2Bfilter=self.filer(tmp_mat,self.input)
        tmp_res=tf.matmul(X2Bfilter,self.w[index])
        tmp_vec=self.vec(tmp_res)
        return tmp_res,tmp_vec
    def predict(self,input):
        tmp_res=input
        vec_list=[]
        for index in range(len(self.layerList)):
            tmp_res,tmp_vec=self.forward(tmp_res,index)
            vec_list.append(tmp_vec)
        out_res=vec_list[0]
        for i in range(1,len(vec_list)):
            out_res=tf.concat([out_res,vec_list[i]],axis=1)
        return out_res


