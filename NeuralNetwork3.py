import numpy as np
import csv
import time
from sys import argv

def softmax(inputs):
    '''
    This activation function is for the output layer to choose from

    :param inputs: An array of nodes
    :return: an array of nodes
    '''
    return np.exp(inputs) / np.sum(np.exp(inputs), axis=0, keepdims=True)

def sigmoid(inputs):
    '''
    This activation functoin is for the other hidden layer

    :param input: An array of floating number
    :return: An array floating number
    '''
    output = np.array(inputs, dtype=np.float128)
    return 1.0 / (1 + np.exp(-1 * output))

def derivative(inputs):
    '''
    this is the derivative of sigmoid and softmax function
    :param inputs:
    :return:
    '''
    output = np.array(inputs, dtype=np.float128)
    sig = 1.0 / (1 + np.exp(-1 * output ))
    return sig * (1 - sig)

#####initializer function
def translate(inputs,parse = True):
    '''
    an array of string to array of int

    :param inputs: an array os string
    :return: an array of int
    '''

    temp = []

    for input in inputs:
        input_int = [float(i) for i in input]

        temp.append(input_int)
    output = np.array(temp)
    if parse:
        parsing = int(output.shape[0]/6)
        return output[:parsing,:]
    else:
        return  output

def split(input, batch):
    '''
    Split the data into batches

    :param input: the dataset
    :param batch: number of images in one set
    :return: input that have been split into batches
    '''
    output = []

    l = len(input)
    for n in range(0, l, batch):
        transpose = np.array(input[n:n+batch, :],dtype= np.float32).T
        output.append(transpose.tolist())
    return np.array(output)

def prepare_data(batch , image ="./train_image.csv",label= './train_label.csv', test='./test_image.csv'):
    training_image_file = open(image,"r")
    images = np.array(translate(csv.reader(training_image_file),False)) / 255
    images_batch = split(images, batch)

    label_file = open(label,"r")
    labels = translate(csv.reader(label_file),False)
    print("This is labels: ",labels.shape)
    desired_labels = []

    for label in labels:
        desired_label = label[0]
        desired_labels.append([float(1) if (x == desired_label) else float(0) for x in range(10)])
    desired = split(np.array(desired_labels),batch)

    test_image_file = open(test,"r")
    test_images = np.array(translate(csv.reader(test_image_file),False)).T / 255
    return images_batch,desired,test_images

def initialize_parameters(layer_dims):
    L = len(layer_dims)
    for l in range(1,L):
        wb["w"+str(l-1)] = np.random.randn(layer_dims[l], layer_dims[l - 1])
        wb["b"+str(l-1)] = np.zeros((layer_dims[l], 1))
    return wb

#Start Forward prop
def feed_for(wb,train_images):
    forward_param = {}
    forward_param['z'+str(0)]= np.dot(wb["w0"], train_images)+wb["b0"]
    forward_param['a'+str(0)]= sigmoid(forward_param['z'+str(0)])

    for i in range(1,len(weight_layer)-1):
        forward_param['z'+str(i)]= np.dot(wb["w"+str(i)],forward_param['a'+str(i-1)] )+wb["b"+str(i)]
        if i == len(weight_layer)-2:
            forward_param['a' + str(i)] = softmax(forward_param['z' + str(i)])
        else:
            forward_param['a' + str(i)] = sigmoid(forward_param['z' + str(i)])
    return forward_param

#####compute lost
def loss_function(forward_param):
    cost = -np.sum((train_label* np.log(forward_param["a"+str(len(weight_layer)-2)])),
                   axis=0, keepdims=True)
    return np.sum(cost)/ train_images.shape[1]

####backward propagation
def back_prop(wb, forward_param):
    back_param = {}

    back_param["dz" + str(len(weight_layer)-2)] = (forward_param["a" + str(len(weight_layer)-2)]
                                                   -train_label)/train_images.shape[1]
    for i in range(len(weight_layer)-3,-1,-1):
        back_param["da" + str(i)]= np.dot(wb["w"+str(i+1)].T, back_param["dz"+str(i+1)])
        back_param["dz" + str(i)] = back_param["da" + str(i)]*derivative(forward_param["z"+str(i)])

    back_param['dw'+str(0)] = np.dot(back_param['dz'+str(0)], train_images.T)
    back_param['db'+str(0)] = np.sum(back_param['dz'+str(0)],axis= 1,keepdims= True)
    for j in range(1, len(weight_layer)-1):
        back_param['dw' + str(j)] = np.dot(back_param['dz' + str(j)], forward_param['a'+str(j-1)].T)
        back_param['db' + str(j)] = np.sum(back_param['dz' + str(j)], axis=1, keepdims=True)
    return back_param

####### Update weight and bias
def update(back_param, learning_rate):
    for layer in range(len(weight_layer)-1):

        wb['w'+str(layer)] = wb['w'+str(layer)] - learning_rate * back_param["dw"+str(layer)]
        wb['b' + str(layer)] = wb['b' + str(layer)] - learning_rate * back_param["db" + str(layer)]

    return wb

####prediction
def predict(wb,test_images):
    predictions = []
    # accuracy =[]

    ##### Delete later####
    # label_file = open("./test_label.csv")
    # labels = translate(csv.reader(label_file),False)
    #############
    forward_param = feed_for(wb, test_images)
    predict = np.around(forward_param['a'+str(len(weight_layer)-2)], decimals=2)

    for image in range(test_images.shape[1]):
        predicted_image = np.argmax(predict[:,image])
        # label = labels[image][0]
        # if predicted_image == label:
        #     accuracy.append(1)
        predictions.append([predicted_image])
    # print("This is the accuracy of the network: ", np.sum(accuracy)/test_images.shape[1])

    return np.array(predictions)

###### initializing variable
start_time = time.time()
wb ={}
weight_layer= [784, 200, 100, 50,10]
initialize_parameters(weight_layer)
batch = 10

if len(argv)!=4:
    train_ima, train_la, test_images = prepare_data(batch)
else:
    train_ima, train_la, test_images = prepare_data(batch,argv[1],argv[2],argv[3])


epoch = 60
learning_rate = 0.5

print(train_la.shape)


for l in range(0, epoch):
    print("This is epoch: ", l)
    for bat in range(train_ima.shape[1]):
        train_images = train_ima[bat]
        train_label = train_la[bat]
        print(train_label.shape)
        exit()
        forward_param = feed_for(wb, train_images)
        cost = loss_function(forward_param)
        back_param = back_prop(wb,forward_param)
        update(back_param, learning_rate)
        print("Loss: ", cost)
    print()

predictions = predict(wb,test_images)
##this is csv write
file = open("./test_predictions.csv",'w')
csvwriter = csv.writer(file)

# writing the data rows
csvwriter.writerows(predictions)
# np.savetxt('test_predictions.csv',predictions,delimiter=',', fmt='%d')

print("--- %s seconds ---" % ((time.time() - start_time)/60))