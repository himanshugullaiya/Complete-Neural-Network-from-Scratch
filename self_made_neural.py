
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class neural_network:
      def __init__(self, layers_size): 
           
            self.no_layers = len(layers_size)
            self.weights = []
            self.baises = []
            self.errors = []
            self.delta = []
            
            for x in range(1,self.no_layers):
                  self.weights.append(np.random.uniform(size = (layers_size[x-1],layers_size[x])))
                  self.baises.append(np.random.uniform(size = (1,layers_size[x])))
      
      def square(self,x):
            return x**2
            
      def relu(self,array):
            x,y = array.shape
            for a in range(x):
                  for b in range(y):
                        if array[a][b] < 0:
                              array[a][b] = 0
            return array
      
      def sigmoid(self,a):
            return 1/(1+np.e**(-a))
      
      def derv(self,a):
            return self.sigmoid(a)*(1-self.sigmoid(a))
#            return a*(1-a)
      
      def forward_feed(self, input_layer):
            
            self.f_all_layers = []
            self.f_all_layers.append(input_layer)
            
            
            for x in range(self.no_layers-1):
                  temp_hidden = np.dot(self.f_all_layers[x], self.weights[x])+ self.baises[x]
                  
                  if x == self.no_layers-2:
                        hidden_activation = self.sigmoid(temp_hidden)
                  else:
                        hidden_activation = self.relu(temp_hidden)
                  self.f_all_layers.append(hidden_activation)
      
      def backpropagate(self, out_layer):
            self.train_output = out_layer
            self.errors.clear()
            self.delta.clear()
#            output
            output_error = self.train_output - self.f_all_layers[-1]
            output_slope = self.derv(self.f_all_layers[-1])
            output_delta = output_error*output_slope
            self.errors.append(output_error)
            self.delta.append(output_delta)
            
#            hidden layers     
            for x in range(self.no_layers-2,0,-1):
                  try:
                        temp_hid_error = np.dot(self.delta[0], np.transpose(self.weights[x]))
                  except:
                        temp_hid_error = np.dot(self.delta[x], np.transpose(self.weights[x]))
                  slope_hidden = self.derv(self.f_all_layers[x])
                  temp_hid_delta = temp_hid_error*slope_hidden
                  self.errors.insert(0, temp_hid_error)
                  self.delta.insert(0, temp_hid_delta)

            
      def update(self):
      
            try:
                  for x in range(self.no_layers-2,-1,-1):
                        self.weights[x] += np.dot(np.transpose(self.f_all_layers[x]), self.delta[x])*self.alpha
                        self.baises[x] += np.sum(self.delta[x], axis = 0)*self.alpha
            except:
                  print(x)
            
      def cost(self, error):
            sqrd_error = self.square(error)
            no_inputs = error.shape[0]
            no_outputs = error.shape[1]
            sqrd_error = (np.sum(sqrd_error, axis = 0)).reshape(1,-1)
            sqrd_error = np.sum(sqrd_error, axis = 1)
            return (sqrd_error/(2*no_outputs*no_inputs)).item()
            
      
      def predict(self, test_input):
            self.forward_feed(test_input)
            return self.f_all_layers[-1]
                  
      def test_error(self, test_output):
            output_error = test_output - self.f_all_layers[-1]
            return self.cost(output_error)
            
      def driver(self, input_layer, output_layer, epochs, batch_size):
            self.alpha = 0.01
#            for x in range(1):
#                  self.forward_feed(input_layer)
#                  self.backpropagate(output_layer)
#                  self.update()
#                  print(f"Epoch: {x+1}/{epochs} Error: {self.cost(self.errors[-1])}")
           
            iterations = math.floor(input_layer.shape[0]/batch_size)            
            input_layer_1  = input_layer[0:iterations*batch_size, :]
            output_layer_1 = output_layer[0:iterations*batch_size, :]
            
            left = input_layer.shape[0]-iterations*batch_size
            
            input_layer_2  =  input_layer[iterations*batch_size:,:]
            output_layer_2 =  output_layer[iterations*batch_size:,:]
            
            for y in range(0,epochs): 
                  for x in range(1,iterations+1): 
                        self.forward_feed(input_layer_1[(x-1)*batch_size : x*batch_size, : ])
                        self.backpropagate(output_layer_1[(x-1)*batch_size : x*batch_size, : ])
                        self.update()
                  
                  try:
                        for x in range(left):
                              self.forward_feed(input_layer_2)
                              self.backpropagate(output_layer_2)
                  except:
                        pass
                  print(f"Epoch: {y+1}/{epochs} Error: {self.cost(self.errors[-1])}")
