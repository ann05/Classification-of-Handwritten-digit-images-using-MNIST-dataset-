# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:47:00 2016

@author: ANMISHA
"""

import cPickle
import gzip
import numpy as np
import random as rand
import glob
from PIL import Image
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()
temp1=0
#print("0",np.shape(training_data[0]))
#print("1",np.shape(training_data[1]))
#def Logistic(X,Y):
#print np.shape(validation_data[0])

#---------Logistic--------------

#print("T",T)
def Run2(D,N):
    #N=50000
    X=np.zeros((N,784),dtype='float')
    X=D[0]
    #print X
    T=np.zeros((N,1),dtype='float')
    T=D[1]
    Wi=np.zeros((784,10),dtype='float')
    for row in range(0,784):
        for col in range(0,10):
            Wi[row][col]=rand.random()
    #print Wi
    eta=1
    print "learning rate=",eta
    A=np.zeros((N,10),dtype='float')
    B=np.ones((N,10),dtype='int')
    #for i in range(0,N):
    temp=X.dot(Wi)
    A=temp + B
    #X1=np.array(X)
    #Xt=np.zeros((784,N),dtype='float')
    #print("X ",X[40])
    Xt=np.transpose(X)
    #print("Xt ",(Xt[:,40]))
    Tnew=np.zeros((N,10),dtype='float')
    for row in range(0,N):
           # for col in range(0,10):
             x=T[row]
             Tnew[row][int(x)]=1
    #print np.shape(temp)
    Y=np.zeros((N,10),dtype='float')
    for j in range(0,100):
        temp=X.dot(Wi)
        A=temp + B
        for row in range(0,N):
            temp1=0
            for i in range(0,10):
                temp1=temp1+np.exp(A[row][i])
            #print ("t1 ",temp1)
            for col in range(0,10):
                t=(np.exp(A[row][col]))
                #print ("t ",t)
                x=(t/temp1)
                Y[row][col]=x
        #print(" Y",Y)
        #print T
        
        #print Tnew
        temp2=Y-Tnew
        temp3=Xt.dot(temp2)
        gradient=temp3
        #print temp3
        gradient2=(eta*gradient)/N
        #print gradient2
        Wn=Wi-gradient2
        #print("Wi",Wi[40])
        #print Wn[40]
        Wi=Wn
        
    Nwrong=0
    Tfinal=np.zeros((N,1),dtype='float')
    for i in range(0,N):
        Tfinal[i]=np.argmax(Y[i])
   # print Tfinal
    #print T
    for i in range(0,N):
        if Tfinal[i]==T[i]:
            Nwrong=Nwrong+1
    #print("N", Nwrong)
    t=Nwrong*100
    t=t/N
    print "Performance for Logistic Regression ",t
    
    #-------------Single Layer Neural Network---------------
    Ws=np.zeros((784,100),dtype='float')
    for row in range(0,784):
        for col in range(0,100):
            Ws[row][col]=rand.random()*0.1
    Bs=np.ones((1,100),dtype='int')
    Bk=np.ones((1,10),dtype='int')
    temp4=np.zeros((N,100),dtype='float')
    Wkj=np.zeros((100,10),dtype='int')
    Ys=np.zeros((N,10),dtype='float')
    Yt=np.zeros((1,10),dtype='float')
    t11=np.zeros((784,1),dtype='float')
    eta=0.01
    print "learning rate2",eta
    for i in range (0,100):
        for j in range(0,10):
                Wkj[i][j]=np.random.rand()*0.1
                
    for loop in range(0,5):
        for row in range(0,N):
            temp6=0
            temp4=X[row].dot(Ws)
            temp4=temp4+Bs
            #print np.shape(temp4),np.shape(Bs)
            Z=(1/(1+np.exp(-temp4)))
            #print np.shape(Z)
            #print temp4
            #print(h,np.shape(h))
            
            
            temp5=Z.dot(Wkj)
            A2=temp5+Bk
            #print A2,np.shape(A2)
            """
           
            for col in range(0,10):
                t1=(np.exp(A[row][col]))
            """
            for i in range(0,10):
                    temp6=temp6+np.exp(A2[0][i])
                #print ("t1 ",temp1)
            for col in range(0,10):
                t1=(np.exp(A2[0][col]))
                #Ys[row][col]=A2[row][col]
                    #print ("t ",t)
                x=(t1/temp6)
                Ys[row][col]=x
                Yt[0][col]=x
            #print Yt
            #print ("Ys,Tnew",np.shape(Ys[row]),np.shape(Tnew[row]),np.shape(Yt))   
            dkt=Yt-Tnew[row]
            #print "dkt",np.shape(dkt)
            Zd=np.transpose(Z)
            dk=Zd.dot(dkt)
            #print "dk",np.shape(dk)
            temp7=1-Z
            temp7=np.transpose(temp7)
            #print "temp7",temp7,np.shape(temp7)
            Hd=Z.dot(temp7)
            
            #print "Hd",Hd,np.shape(Hd)   
            #t1=(X[row])
            #print("X",np.shape(X[row]))
            #t1=np.transpose(t1)
            #print("t1",np.shape(t1))
            for j in range(0,784):
                t11[j][0]=X[row][j]
            t2=np.transpose(Wkj)
            t3=t11.dot(dkt)
            dj=Hd*(t3.dot(t2))
            #print "dj",np.shape(dj)
            WNkj=Wkj-eta*dk
            WNs=Ws-eta*dj
            Ws=WNs
            Wkj=WNkj
            
    #print "Ys",np.shape(Ys),Ys
    Nwrong=0
    Tfinal2=np.zeros((N,1),dtype='float')
    for i in range(0,N):
        Tfinal2[i]=np.argmax(Ys[i])
    #print Tfinal2
    #print T
    for i in range(0,N):
        if Tfinal2[i]==T[i]:
            Nwrong=Nwrong+1
    #print("N", Nwrong)
    t=Nwrong*100
    t=t/N
    print "Performance for Single Neural Network ",t

#print "-----------------For MNIST Data-----------"
#print " Training Data "
#Run(training_data,50000)
def Run(D,N):
    #N=50000
    X=np.zeros((N,784),dtype='float')
    X=D[0]
    #print X
    T=np.zeros((N,1),dtype='float')
    T=D[1]
    Wi=np.zeros((784,10),dtype='float')
    for row in range(0,784):
        for col in range(0,10):
            Wi[row][col]=rand.random()
    #print Wi
    eta=1
    print "learning rate",eta
    A=np.zeros((N,10),dtype='float')
    B=np.ones((N,10),dtype='int')
    #for i in range(0,N):
    temp=X.dot(Wi)
    A=temp + B
    #X1=np.array(X)
    #Xt=np.zeros((784,N),dtype='float')
    #print("X ",X[40])
    Xt=np.transpose(X)
    #print("Xt ",(Xt[:,40]))
    Tnew=np.zeros((N,10),dtype='float')
    for row in range(0,N):
           # for col in range(0,10):
             x=T[row]
             Tnew[row][x]=1
    #print np.shape(temp)
    Y=np.zeros((N,10),dtype='float')
    for j in range(0,100):
        temp=X.dot(Wi)
        A=temp + B
        for row in range(0,N):
            temp1=0
            for i in range(0,10):
                temp1=temp1+np.exp(A[row][i])
            #print ("t1 ",temp1)
            for col in range(0,10):
                t=(np.exp(A[row][col]))
                #print ("t ",t)
                x=(t/temp1)
                Y[row][col]=x
        #print(" Y",Y)
        #print T
        
        #print Tnew
        temp2=Y-Tnew
        temp3=Xt.dot(temp2)
        gradient=temp3
        #print temp3
        gradient2=(eta*gradient)/N
        #print gradient2
        Wn=Wi-gradient2
        #print("Wi",Wi[40])
        #print Wn[40]
        Wi=Wn
        
    Nwrong=0
    Tfinal=np.zeros((N,1),dtype='float')
    for i in range(0,N):
        Tfinal[i]=np.argmax(Y[i])
   # print Tfinal
    #print T
    for i in range(0,N):
        if Tfinal[i]==T[i]:
            Nwrong=Nwrong+1
    #print("N", Nwrong)
    t=Nwrong*100
    t=t/N
    print "Performance for Logistic Regression ",t
    
    #-------------Single Layer Neural Network---------------
    Ws=np.zeros((784,100),dtype='float')
    for row in range(0,784):
        for col in range(0,100):
            Ws[row][col]=rand.random()*0.1
    Bs=np.ones((1,100),dtype='int')
    Bk=np.ones((1,10),dtype='int')
    temp4=np.zeros((N,100),dtype='float')
    Wkj=np.zeros((100,10),dtype='int')
    Ys=np.zeros((N,10),dtype='float')
    Yt=np.zeros((1,10),dtype='float')
    t11=np.zeros((784,1),dtype='float')
    eta=0.01
    print "learning rate2",eta
    for i in range (0,100):
        for j in range(0,10):
                Wkj[i][j]=np.random.rand()*0.1
                
    for loop in range(0,5):
        for row in range(0,N):
            temp6=0
            temp4=X[row].dot(Ws)
            temp4=temp4+Bs
            #print np.shape(temp4),np.shape(Bs)
            Z=(1/(1+np.exp(-temp4)))
            #print np.shape(Z)
            #print temp4
            #print(h,np.shape(h))
            
            
            temp5=Z.dot(Wkj)
            A2=temp5+Bk
            #print A2,np.shape(A2)
            """
           
            for col in range(0,10):
                t1=(np.exp(A[row][col]))
            """
            for i in range(0,10):
                    temp6=temp6+np.exp(A2[0][i])
                #print ("t1 ",temp1)
            for col in range(0,10):
                t1=(np.exp(A2[0][col]))
                #Ys[row][col]=A2[row][col]
                    #print ("t ",t)
                x=(t1/temp6)
                Ys[row][col]=x
                Yt[0][col]=x
            #print Yt
            #print ("Ys,Tnew",np.shape(Ys[row]),np.shape(Tnew[row]),np.shape(Yt))   
            dkt=Yt-Tnew[row]
            #print "dkt",np.shape(dkt)
            Zd=np.transpose(Z)
            dk=Zd.dot(dkt)
            #print "dk",np.shape(dk)
            temp7=1-Z
            temp7=np.transpose(temp7)
            #print "temp7",temp7,np.shape(temp7)
            Hd=Z.dot(temp7)
            
            #print "Hd",Hd,np.shape(Hd)   
            #t1=(X[row])
            #print("X",np.shape(X[row]))
            #t1=np.transpose(t1)
            #print("t1",np.shape(t1))
            for j in range(0,784):
                t11[j][0]=X[row][j]
            t2=np.transpose(Wkj)
            t3=t11.dot(dkt)
            dj=Hd*(t3.dot(t2))
            #print "dj",np.shape(dj)
            WNkj=Wkj-eta*dk
            WNs=Ws-eta*dj
            Ws=WNs
            Wkj=WNkj
            
    #print "Ys",np.shape(Ys),Ys
    Nwrong=0
    Tfinal2=np.zeros((N,1),dtype='float')
    for i in range(0,N):
        Tfinal2[i]=np.argmax(Ys[i])
    #print Tfinal2
    #print T
    for i in range(0,N):
        if Tfinal2[i]==T[i]:
            Nwrong=Nwrong+1
    #print("N", Nwrong)
    t=Nwrong*100
    t=t/N
    print "Performance for Single Neural Network ",t

print "-----------------For MNIST Data-----------"
print " Training Data "
Run(training_data,50000)
print "Validation Data "
Run(validation_data,10000)
print "Test Data "
Run(test_data,10000)

usps_x=[]
usps_t=[]
for i in range(0,10):
    for imagedata in glob.glob("Numerals/"+str(i)+"/*.png"):
        img = Image.open(imagedata)
        img = img.resize((28,28))
        usps_x.append(list(img.getdata()))
        usps_t.append(i)
SN=len(usps_x)
uspsD_x=np.zeros((SN,784))
uspsD_t=np.zeros((SN,1))
for i in range(0,SN):
    for j in range(0,784):
        td=((uspsD_x[i][j])/(255))
        uspsD_x[i][j]=1-td
for i in range(0,SN):
    uspsD_t[i][0]=usps_t[i]

#USPS[0]=uspsD_x
#USPS[1]=uspsD_t
print ("----------USPS DATA-----------------")
Run2((uspsD_x,uspsD_t),SN)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#-----------------Convolutional Neural Network-----------
def Run3():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
    def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
    def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='SAME')
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # Densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Add dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print "test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
Run3()