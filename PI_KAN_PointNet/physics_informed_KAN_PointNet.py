# In The Name of God
##### Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular geometries #####

#Author: Ali Kashefi (kashefi@stanford.edu)

##### Citation #####



###### Libraries ######
import os
import csv
import linecache
import math
import timeit
from timeit import default_timer as timer
from operator import itemgetter
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import time
import gc
from torchsummary import summary

###### Device setup ######
if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU available
else:
    device = torch.device("cpu")  # Only CPU available

gc.collect()
torch.cuda.empty_cache()

###### Parameter setup ######
problem_dimension = 2  # (x,y)
variable_number = 4  # (u,v,p,T)
N_boundary = 168+492 # number of points on the boundary
outter_surface_n = 492 # number of points on the outer surface

SCALE = 0.5 # To control the network size
poly_degree = 2 # Polynomial degree of Jacaboi Polynomial
ALPHA = -0.5 # \alpha in Jacaboi Polynomial
BETA = -0.5 # \beta in Jacaboi Polynomial

Learning_rate = 0.0005
Epoch = 2500 # Maximum number of epochs
Nb = 7 # Batch Size (memory sensitive)
J_Loss = 0.0001

density = 1.0 # fluid density
viscosity = np.power(2.0,1.5)*np.power(10.0,-2.5) # fluid viscosity
kappa =  np.power(2.0,1.5)*np.power(10.0,-2.5) # thermal conductivity

###### Data loading and data preparation ######

BC_list = [] #point number on boundary
full_list = [] #point number on the whole domain 
interior_list = [] #interior nodes without full, BC, sparse

Data = np.load('Data.npy')
data, num_points, _ = Data.shape

X_train = Data[:, :, :problem_dimension]
CFD_train = Data[:, :, problem_dimension:problem_dimension + variable_number]

cfd_u = CFD_train[:, :, 0].reshape(-1)
cfd_v = CFD_train[:, :, 1].reshape(-1)
cfd_p = CFD_train[:, :, 2].reshape(-1)
cfd_T = CFD_train[:, :, 3].reshape(-1)

###### Sensor locations and setup ######
sparse_d = 25
sparse_n = 100+5+sparse_d
sparse_list = [[-1 for i in range(sparse_n)] for j in range(data)] 
BC_list_temperature_inverse = [[-1 for i in range(outter_surface_n)] for j in range(data)] 

###### Shared Kolmogorov-Arnold Networks (KANs) ######
class JacobiKANLayerFirst(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayerFirst, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous()  # shape = (batch_size, num_points, input_dim)
        # x = torch.tanh(x)  # if the input data is already between [-1, 1], no need to normalize x to [-1, 1]

        # Initialize Jacobi polynomial tensors
        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        # Compute the Jacobi interpolation
        jacobi = jacobi.permute(0, 2, 3, 1)  # shape = (batch_size, input_dim, degree + 1, num_points)
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs)  # shape = (batch_size, output_dim, num_points)

        return y

###### Shared Kolmogorov-Arnold Networks (KANs) ######
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous()  # shape = (batch_size, num_points, input_dim)
        x = torch.tanh(x)  # Normalize x to [-1, 1]

        # Initialize Jacobi polynomial tensors
        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        # Compute the Jacobi interpolation
        jacobi = jacobi.permute(0, 2, 3, 1)  # shape = (batch_size, input_dim, degree + 1, num_points)
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs)  # shape = (batch_size, output_dim, num_points)

        return y

###### PointNet with shared KANs ######
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=1.0, Alpha=1.0, Beta=1.0):
        super(PointNetKAN, self).__init__()

        #Shared KAN (64, 64)
        self.jacobikan1 = JacobiKANLayerFirst(input_channels, int(64 * scaling), poly_degree, Alpha, Beta)
        self.jacobikan2 = JacobiKANLayer(int(64 * scaling), int(64 * scaling), poly_degree, Alpha, Beta)

        #Shared KAN (64, 128, 1024)
        self.jacobikan3 = JacobiKANLayer(int(64 * scaling), int(64 * scaling), poly_degree, Alpha, Beta)
        self.jacobikan4 = JacobiKANLayer(int(64 * scaling), int(128 * scaling), poly_degree, Alpha, Beta)
        self.jacobikan5 = JacobiKANLayer(int(128 * scaling), int(1024 * scaling), poly_degree, Alpha, Beta)

        #Shared KAN (512, 256, 128)
        self.jacobikan6 = JacobiKANLayer(int(1024 * scaling) + int(64 * scaling), int(512 * scaling), poly_degree, Alpha, Beta)
        self.jacobikan7 = JacobiKANLayer(int(512 * scaling), int(256 * scaling), poly_degree, Alpha, Beta)
        self.jacobikan8 = JacobiKANLayer(int(256 * scaling), int(128 * scaling), poly_degree, Alpha, Beta)

        #Shared KAN (128, output_channels)
        self.jacobikan9 = JacobiKANLayer(int(128 * scaling), int(128 * scaling), poly_degree, Alpha, Beta)
        self.jacobikan10 = JacobiKANLayer(int(128 * scaling), output_channels, poly_degree, Alpha, Beta)

        #Batch Normalization
        self.bn1 = nn.BatchNorm1d(int(64 * scaling))
        self.bn2 = nn.BatchNorm1d(int(64 * scaling))
        self.bn3 = nn.BatchNorm1d(int(64 * scaling))
        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))
        self.bn6 = nn.BatchNorm1d(int(512 * scaling))
        self.bn7 = nn.BatchNorm1d(int(256 * scaling))
        self.bn8 = nn.BatchNorm1d(int(128 * scaling))
        self.bn9 = nn.BatchNorm1d(int(128 * scaling))

    def forward(self, x):

        # Shared KAN (64, 64)
        x = self.jacobikan1(x)
        x = self.bn1(x)
        x = self.jacobikan2(x)
        x = self.bn2(x)

        local_feature = x

        # Shared KAN (64, 128, 1024)
        x = self.jacobikan3(x)
        x = self.bn3(x)
        x = self.jacobikan4(x)
        x = self.bn4(x)
        x = self.jacobikan5(x)
        x = self.bn5(x)

        # Max pooling to get the global feature
        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, num_points)

        # Concatenate local and global features
        x = torch.cat([local_feature, global_feature], dim=1)

        # Shared KAN (512, 256, 128)
        x = self.jacobikan6(x)
        x = self.bn6(x)
        x = self.jacobikan7(x)
        x = self.bn7(x)
        x = self.jacobikan8(x)
        x = self.bn8(x)

        # Shared KAN (128, output_channels)
        x = self.jacobikan9(x)
        x = self.bn9(x)
        x = self.jacobikan10(x)

        return x

###### Functions for data visualization and error calculations ######

def plotSolutions2DPoint(x,y,variable,index,name,title):    
    marker_size= 1.0 
    plt.scatter(x, y, marker_size, variable, cmap='jet')
    cbar= plt.colorbar()
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=6)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name+'.png',dpi=300)
    plt.savefig(name + '.pdf')
    #plt.savefig(name+'.eps')    
    plt.clf()
    #plt.show()

def plotCost(Y,name,title):
    plt.plot(Y)
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(name+'.png',dpi = 300,bbox_inches='tight')
    plt.savefig(name+'.eps',bbox_inches='tight')
    plt.clf()
    #plt.show()

def relative_l2_norm(prediction, truth):
    
    l2_norm_diff = np.linalg.norm(prediction - truth)
    l2_norm_v1 = np.linalg.norm(truth)
       
    return l2_norm_diff / l2_norm_v1

###### Helper functions for sensor setup ######
def find_on_boundary(x_i,y_i,data_number,find_value):
    
    call = -1
    for index in range(0,N_boundary):  
        if np.sqrt(np.power(X_train[data_number][index][0]-x_i,2.0) + np.power(X_train[data_number][index][1]-y_i,2.0)) < np.power(10.0,-11.0): #np.power(10.0,-4.0):
            call = index
            break
    return call

def find_on_interior(x_i,y_i,data_number,find_value):
    call = -1
    for index in range(N_boundary,num_points):  
        if np.sqrt(np.power(X_train[data_number][index][0]-x_i,2.0) + np.power(X_train[data_number][index][1]-y_i,2.0)) < np.power(10.0,find_value): #np.power(10.0,-4.0):
            call = index
            break
    return call


def problemSet_nonuniform_grid():
    Lambda_pde = 1.0
    Lambda_bc = 1.0
    Lambda_sparse = 1.0

    for i in range(N_boundary):
        BC_list.append(i)

    #Number of points on the boundary should be fixed over all geometries
    #Number of points on the inner surface should be fixed over all geometries
    count_outter_surface = 0
    for j in range(data):
        count_outter_surface = 0     
        for i in range(N_boundary):
            if np.sqrt(np.square(0.0-X_train[j][i][0]) + np.square(0.0*np.pi-X_train[j][i][1])) > (0.8):
                BC_list_temperature_inverse[j][count_outter_surface] = i                       
                count_outter_surface += 1
                if (count_outter_surface + 1) > outter_surface_n:
                    break  
       
    for i in range(data):
        for j in range(outter_surface_n):
            if BC_list_temperature_inverse[i][j] == -1:
                print("warning: -1 exists in temperature inverse list")

    for i in range(num_points):
        full_list.append(i)
    
    # structured sensors
    find_value_store = []
    for ci in range(45):
        find_value_store.append(-1.20)
    for ci in range(50):
        find_value_store.append(-1.20)
    for ci in range(40):
        find_value_store.append(-1.25)
        
    loc = np.zeros((data,8),dtype=float)
    for i in range(data):
        loc[i][0] = -0.75 
        loc[i][1] = -0.6 
        loc[i][2] = 0.6 
        loc[i][3] = 0.83 
        loc[i][4] = -0.85 
        loc[i][5] = -0.6 
        loc[i][6] = 0.6 
        loc[i][7] = 0.85 
        
    x_c = 0.0 
    y_c = 0.0 
    var_x = 0.1 
    var_y = 0.05
    step_x = 0.12
    step_y = 0.12
    for i in range(data):
        cal = 0
        conuter_s = 0
        x_sparse = np.zeros((sparse_n-sparse_d),dtype=float)
        y_sparse = np.zeros((sparse_n-sparse_d),dtype=float)
        xd_sparse = np.zeros((sparse_d),dtype=float)
        yd_sparse = np.zeros((sparse_d),dtype=float)
        boundary_nodes = np.zeros(shape=(N_boundary - outter_surface_n,2),dtype=float)
        for j in range(N_boundary):
            if np.sqrt(np.square(0.0-X_train[i][j][0]) + np.square(0.0-X_train[i][j][1])) > (0.8):
                continue
            boundary_nodes[cal][0] = 1.0*j
            boundary_nodes[cal][1] = np.arctan2(X_train[i][j][0]-0.0,X_train[i][j][1]-0.0)
            cal += 1
        boundary_nodes = boundary_nodes[np.argsort(boundary_nodes[:, 1])]
        for j in range(13):
            x_sparse[conuter_s] = -1 + (j+2)*step_x 
            y_sparse[conuter_s] = loc[i][0] #4.0
            conuter_s += 1
        for j in range(5):
            x_sparse[conuter_s] = -1 + 0.5 + (j+2)*step_x 
            y_sparse[conuter_s] = loc[i][1] #4.5
            conuter_s += 1
        for j in range(13):
            x_sparse[conuter_s] = -1 + (j+2)*step_x 
            y_sparse[conuter_s] = loc[i][2] #8.2
            conuter_s += 1
        for j in range(13):
            x_sparse[conuter_s] = -1 + (j+2)*step_x 
            y_sparse[conuter_s] = loc[i][3] #9.0
            conuter_s += 1
        for j in range(12):
            x_sparse[conuter_s] = loc[i][4] #3.6 
            y_sparse[conuter_s] = -1 + (j+1)*step_y*1.2
            conuter_s += 1
        for j in range(6):
            x_sparse[conuter_s] = loc[i][5] #4.2 
            y_sparse[conuter_s] = -1 + (j+2)*step_y*1.7
            conuter_s += 1
        for j in range(6):
            x_sparse[conuter_s] = loc[i][6] #8.2 
            y_sparse[conuter_s] = -1 + (j+2)*step_y*1.7
            conuter_s += 1
        for j in range(12):
            x_sparse[conuter_s] = loc[i][7] #9.0 
            y_sparse[conuter_s] = -1 + (j+1)*step_y*1.2
            conuter_s += 1
        
        reminder = sparse_n - 12-12-13-13-13-6-6-5 - sparse_d
        counter_d = 0
        for j in range(reminder):
            xxx = X_train[i][int(boundary_nodes[int(6.8*(j))][0])][0]
            yyy = X_train[i][int(boundary_nodes[int(6.8*(j))][0])][1]
            if xxx - x_c > 0 and yyy - y_c > 0:
                x_sparse[conuter_s] = xxx + var_x
                y_sparse[conuter_s] = yyy + var_y
                conuter_s += 1
                xd_sparse[counter_d] = xxx 
                yd_sparse[counter_d] = yyy
                counter_d += 1
                if conuter_s > sparse_n - 1:
                    break
            if xxx - x_c >= 0 and yyy - y_c <= 0:
                x_sparse[conuter_s] = xxx + var_x
                y_sparse[conuter_s] = yyy - var_y
                conuter_s += 1
                xd_sparse[counter_d] = xxx 
                yd_sparse[counter_d] = yyy
                counter_d += 1
                if conuter_s > sparse_n  - 1:
                    break
            if xxx - x_c <= 0 and yyy - y_c <= 0:
                x_sparse[conuter_s] = xxx - var_x
                y_sparse[conuter_s] = yyy - var_y
                conuter_s += 1
                xd_sparse[counter_d] = xxx 
                yd_sparse[counter_d] = yyy
                counter_d += 1
                if conuter_s > sparse_n  - 1:
                    break
            if xxx - x_c < 0 and yyy - y_c > 0:
                x_sparse[conuter_s] = xxx - var_x
                y_sparse[conuter_s] = yyy + var_y
                conuter_s += 1
                xd_sparse[counter_d] = xxx 
                yd_sparse[counter_d] = yyy
                counter_d += 1
                if conuter_s > sparse_n  - 1:
                    break
        for j in range(sparse_n-sparse_d):
            sparse_list[i][j] = find_on_interior(x_sparse[j],y_sparse[j],i,find_value_store[i])
        
        corban = 0    
        for j in range(sparse_n-sparse_d,sparse_n):
            sparse_list[i][j] = find_on_boundary(xd_sparse[corban],yd_sparse[corban],i,find_value_store[i])
            corban += 1
            
        #### Plotting ####
        x_outter_surface = np.zeros((outter_surface_n),dtype=float)
        y_outter_surface = np.zeros((outter_surface_n),dtype=float)
        x_boundary0 = np.zeros((N_boundary),dtype=float)
        y_boundary0 = np.zeros((N_boundary),dtype=float)

        for j in range(N_boundary):
            x_boundary0[j] = X_train[i][j][0]
            y_boundary0[j] = X_train[i][j][1]
        for j in range(outter_surface_n):
            x_outter_surface[j] = X_train[i][BC_list_temperature_inverse[0][j]][0] 
            y_outter_surface[j] = X_train[i][BC_list_temperature_inverse[0][j]][1] 
        
        plt.scatter(x_boundary0,y_boundary0,s=0.5)
        plt.scatter(x_outter_surface,y_outter_surface,s=0.5)
        xx_sparse = np.zeros((sparse_n),dtype=float)
        yy_sparse = np.zeros((sparse_n),dtype=float)
        for j in range(sparse_n):
            xx_sparse[j] = X_train[i][sparse_list[i][j]][0]
            yy_sparse[j] = X_train[i][sparse_list[i][j]][1]
        #plt.scatter(x_sparse,y_sparse,s=5,marker="v") #artifical locations
        plt.scatter(xx_sparse,yy_sparse,s=5,marker="v") #exact locations
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('sparsity'+str(i)+'.png',dpi=300)
        #plt.savefig('sparsity.eps')    
        plt.clf()
    
    for i in range(data):
        for j in range(len(x_sparse)):
            if sparse_list[i][j] == -1:
                print("warning: -1 exists in sparse list")
                print(i)
                
    for i in range(num_points):
        if i in BC_list:
            continue
        interior_list.append(i)

##################################

def computeRelativeL2OnSurface(X,Tp,index):

    T_truth = 1.0
    
    Nu_con = 0
    for i in range(N_boundary):
        if np.sqrt(np.square(0.0*np.pi-X_train[index][i][0]) + np.square(0.0*np.pi-X_train[index][i][1])) > (0.8):
            Nu_con += 1
    
    Nu_con = N_boundary - Nu_con
    Nu_surface = np.random.normal(size=(Nu_con, 4)) #x,y,T,alpha
    Nu_con = 0

    for i in range(N_boundary):
        if np.sqrt(np.square(0.0*np.pi-X_train[index][i][0]) + np.square(0.0*np.pi-X_train[index][i][1])) > (0.8):
            continue
            
        Nu_surface[Nu_con][0] = X_train[index][i][0]
        Nu_surface[Nu_con][1] = X_train[index][i][1]
        Nu_surface[Nu_con][2] = Tp[i]
        Nu_surface[Nu_con][3] = np.arctan2(Nu_surface[Nu_con][0]-0*np.pi,Nu_surface[Nu_con][1]-0*np.pi)
        Nu_con += 1

    sum1=0
    for i in range(Nu_con - 1):
        sum1 += np.square(Nu_surface[i][2]-T_truth)
        
    return np.sqrt(sum1/(N_boundary-outter_surface_n))

def compute_T_surface(X,Tp,index):
    
    Nu_con = 0
    for i in range(N_boundary):
        if np.sqrt(np.square(0.0*np.pi-X[index][i][0]) + np.square(0.0*np.pi-X[index][i][1])) > (0.8):
            Nu_con += 1
    
    Nu_con = N_boundary - Nu_con
    Nu_surface = np.random.normal(size=(Nu_con, 4)) #x,y,T,alpha
    Nu_con = 0

    for i in range(N_boundary):
        if np.sqrt(np.square(0.0*np.pi-X[index][i][0]) + np.square(0.0*np.pi-X[index][i][1])) > (0.8):
            continue
            
        Nu_surface[Nu_con][0] = X[index][i][0]
        Nu_surface[Nu_con][1] = X[index][i][1]
        Nu_surface[Nu_con][2] = Tp[i]
        Nu_surface[Nu_con][3] = np.arctan2(Nu_surface[Nu_con][1]-0.0, Nu_surface[Nu_con][0]-0.0)
        Nu_con += 1

    Nu_surface = Nu_surface[np.argsort(Nu_surface[:, 3])]
    
    information = open("Temperature"+str(index+1)+".txt", "w")
    for i in range(Nu_con-1):        
        information.write(str(Nu_surface[i][3])+'  '+str(Nu_surface[i][2]))
        information.write('\n')
    information.close()

###### Problem Set Up ######
problemSet_nonuniform_grid()

###### Physics informed loss function ######
def TheLossEfficient(model,X,pose_BC,pose_sparse,pose_interior,pose_BC_temperature_p,pose_BC_p,pose_sparse_p,pose_interior_p):

    Y = model(X)

    pose_BC = torch.tensor(pose_BC, dtype=torch.long, device=Y.device)
    pose_sparse= torch.tensor(pose_sparse, dtype=torch.long, device=Y.device)
    pose_interior = torch.tensor(pose_interior, dtype=torch.long, device=Y.device)
    pose_BC_temperature_p = torch.tensor(pose_BC_temperature_p, dtype=torch.long, device=Y.device)
    pose_BC_p = torch.tensor(pose_BC_p, dtype=torch.long, device=Y.device)
    pose_sparse_p = torch.tensor(pose_sparse_p, dtype=torch.long, device=Y.device)
    pose_interior_p = torch.tensor(pose_interior_p, dtype=torch.long, device=Y.device)

    u_in = Y[:, 0, :].contiguous().view(-1)[pose_interior_p]
    v_in = Y[:, 1, :].contiguous().view(-1)[pose_interior_p]
    T_in = Y[:, 3, :].contiguous().view(-1)[pose_interior_p]

    u_boundary = Y[:, 0, :].contiguous().view(-1)[pose_BC_p]
    v_boundary = Y[:, 1, :].contiguous().view(-1)[pose_BC_p]
    T_boundary = Y[:, 3, :].contiguous().view(-1)[pose_BC_temperature_p]

    u_sparse = Y[:, 0, :].contiguous().view(-1)[pose_sparse_p]
    v_sparse = Y[:, 1, :].contiguous().view(-1)[pose_sparse_p]
    p_sparse = Y[:, 2, :].contiguous().view(-1)[pose_sparse_p]
    T_sparse = Y[:, 3, :].contiguous().view(-1)[pose_sparse_p]

    du = torch.autograd.grad(Y[:, 0, :], X, grad_outputs=torch.ones_like(Y[:, 0, :]), create_graph=True)[0]
    dv = torch.autograd.grad(Y[:, 1, :], X, grad_outputs=torch.ones_like(Y[:, 1, :]), create_graph=True)[0]
    dp = torch.autograd.grad(Y[:, 2, :], X, grad_outputs=torch.ones_like(Y[:, 2, :]), create_graph=True)[0]
    dT = torch.autograd.grad(Y[:, 3, :], X, grad_outputs=torch.ones_like(Y[:, 3, :]), create_graph=True)[0]
    
    du_dx_in = du[:, 0, :].contiguous().view(-1)[pose_interior_p]
    du_dy_in = du[:, 1, :].contiguous().view(-1)[pose_interior_p]
    dv_dx_in = dv[:, 0, :].contiguous().view(-1)[pose_interior_p]
    dv_dy_in = dv[:, 1, :].contiguous().view(-1)[pose_interior_p]
    dp_dx_in = dp[:, 0, :].contiguous().view(-1)[pose_interior_p]
    dp_dy_in = dp[:, 1, :].contiguous().view(-1)[pose_interior_p]
    dT_dx_in = dT[:, 0, :].contiguous().view(-1)[pose_interior_p]
    dT_dy_in = dT[:, 1, :].contiguous().view(-1)[pose_interior_p]

    d2u_x = torch.autograd.grad(du[:, 0, :], X, grad_outputs=torch.ones_like(du[:, 0, :]), create_graph=True)[0]
    d2u_y = torch.autograd.grad(du[:, 1, :], X, grad_outputs=torch.ones_like(du[:, 1, :]), create_graph=True)[0]    
    
    d2v_x = torch.autograd.grad(dv[:, 0, :], X, grad_outputs=torch.ones_like(dv[:, 0, :]), create_graph=True)[0]
    d2v_y = torch.autograd.grad(dv[:, 1, :], X, grad_outputs=torch.ones_like(dv[:, 1, :]), create_graph=True)[0]

    d2T_x = torch.autograd.grad(dT[:, 0, :], X, grad_outputs=torch.ones_like(dT[:, 0, :]), create_graph=True)[0]
    d2T_y = torch.autograd.grad(dT[:, 1, :], X, grad_outputs=torch.ones_like(dT[:, 1, :]), create_graph=True)[0]

    del du, dv, dT, dp    
    gc.collect()
    torch.cuda.empty_cache()

    d2u_dx2_in = d2u_x[:, 0, :].contiguous().view(-1)[pose_interior_p]
    d2u_dy2_in = d2u_y[:, 1, :].contiguous().view(-1)[pose_interior_p]
    d2v_dx2_in = d2v_x[:, 0, :].contiguous().view(-1)[pose_interior_p]
    d2v_dy2_in = d2v_y[:, 1, :].contiguous().view(-1)[pose_interior_p]
    d2T_dx2_in = d2T_x[:, 0, :].contiguous().view(-1)[pose_interior_p]
    d2T_dy2_in = d2T_y[:, 1, :].contiguous().view(-1)[pose_interior_p]

    del d2u_x, d2u_y, d2v_x, d2v_y, d2T_x, d2T_y    
    gc.collect()
    torch.cuda.empty_cache()

    boundary_u_truth = torch.tensor(cfd_u[pose_BC.cpu()], dtype=torch.float32, device=Y.device)
    boundary_v_truth = torch.tensor(cfd_v[pose_BC.cpu()], dtype=torch.float32, device=Y.device)

    sparse_u_truth = torch.tensor(cfd_u[pose_sparse.cpu()], dtype=torch.float32, device=Y.device)
    sparse_v_truth = torch.tensor(cfd_v[pose_sparse.cpu()], dtype=torch.float32, device=Y.device)
    sparse_T_truth = torch.tensor(cfd_T[pose_sparse.cpu()], dtype=torch.float32, device=Y.device)
    sparse_p_truth = torch.tensor(cfd_p[pose_sparse.cpu()], dtype=torch.float32, device=Y.device)

    r1 = density*(u_in*du_dx_in + v_in*du_dy_in) + dp_dx_in - viscosity*(d2u_dx2_in + d2u_dy2_in)
    r2 = density*(u_in*dv_dx_in + v_in*dv_dy_in) + dp_dy_in - viscosity*(d2v_dx2_in + d2v_dy2_in) - density*1.0*1.0*(T_in - 0.0)
    r3 = du_dx_in + dv_dy_in
    r4 = density*(u_in*dT_dx_in + v_in*dT_dy_in) - (kappa/1.0)*(d2T_dx2_in + d2T_dy2_in)
   
    PDE_cost = torch.mean(r1**2 + r2**2 + r3**2 + r4**2)
    BC_cost = torch.mean((u_boundary - 0.0)**2 + (v_boundary - 0.0)**2) + torch.mean((T_boundary - 0.0)**2)
    Sparse_cost = torch.mean((u_sparse - sparse_u_truth)**2 + (v_sparse - sparse_v_truth)**2 + (p_sparse - sparse_p_truth)**2 + (T_sparse - sparse_T_truth)**2)

    return PDE_cost + Sparse_cost + BC_cost
    #return 100.0*PDE_cost + 100.0*Sparse_cost + BC_cost

###### Model Setup ######
model = PointNetKAN(input_channels=problem_dimension, output_channels=variable_number, scaling=SCALE, Alpha=ALPHA, Beta=BETA)
model = model.to(device)

###### Training and Error Analysis ######
def build_model_Thermal():

    information = open("Loss.txt", "w")
    LOSS_Total = []
    converge_iteration = 0
    
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate) #set epsilon?!
    
    start_ite = time.time()
    t_save = 0

    # Initialize the minimum loss with a large value
    min_loss = float('inf')
    best_model_path = "best_model.pth"

    # training loop
    for epoch in range(Epoch):
        
        temp_cost = 0
        arr = np.arange(data)
        np.random.shuffle(arr)
        
        for sb in range(int(np.ceil(data / Nb))):
            start_idx = int(sb * Nb)
            end_idx = int(min((sb + 1) * Nb, data))
            pointer = arr[start_idx:end_idx]

            group_BC = np.zeros(int(len(pointer)*len(BC_list)), dtype=int)
            group_sparse = np.zeros(int(len(pointer)*sparse_n), dtype=int)
            group_interior = np.zeros(int(len(pointer)*len(interior_list)), dtype=int)

            catch = 0
            for ii in range(len(pointer)):
                for jj in range(len(BC_list)):
                    group_BC[catch] = int(pointer[ii]*num_points + jj)
                    catch += 1

            catch = 0
            for ii in range(len(pointer)):
                for jj in range(sparse_n):
                    group_sparse[catch] = sparse_list[pointer[ii]][jj] + pointer[ii]*num_points
                    catch += 1

            catch = 0
            for ii in range(len(pointer)):
                for jj in range(len(interior_list)):
                    group_interior[catch] = int(pointer[ii]*num_points + len(BC_list) + jj)
                    catch += 1

            group_BC_temperature_p = np.zeros(int(len(pointer)*outter_surface_n), dtype=int)
            group_BC_p = np.zeros(int(len(pointer)*len(BC_list)), dtype=int)
            group_sparse_p = np.zeros(int(len(pointer)*sparse_n), dtype=int)
            group_interior_p = np.zeros(int(len(pointer)*len(interior_list)), dtype=int)

            catch = 0
            for ii in range(len(pointer)):
            #for ii in range(Nb):
                for jj in range(len(BC_list)):
                    group_BC_p[catch] = int(ii*num_points + jj)
                    catch += 1

            catch = 0
            for ii in range(len(pointer)):
            #for ii in range(Nb):
                for jj in range(outter_surface_n):
                    group_BC_temperature_p[catch] = BC_list_temperature_inverse[pointer[ii]][jj] + ii*num_points
                    catch += 1

            catch = 0
            for ii in range(len(pointer)):
            #for ii in range(Nb):
                for jj in range(sparse_n):
                    group_sparse_p[catch] = sparse_list[pointer[ii]][jj] + ii*num_points
                    catch += 1

            catch = 0
            for ii in range(len(pointer)):
            #for ii in range(Nb):
                for jj in range(len(interior_list)):
                    group_interior_p[catch] = int(ii*num_points + len(BC_list) + jj)
                    catch += 1


            X_train_mini = np.take(X_train, pointer[:], axis=0)
            X_train_mini = X_train_mini.transpose(0, 2, 1)
            X_train_mini = torch.tensor(X_train_mini, dtype=torch.float32).requires_grad_(True).to(device)

            optimizer.zero_grad()
            
            # Forward passs
            cost_value = TheLossEfficient(model, X_train_mini, group_BC,group_sparse,group_interior,group_BC_temperature_p,group_BC_p,group_sparse_p,group_interior_p)
            cost_value.backward()
            optimizer.step()
            
            temp_cost_m = cost_value.item()
            
            if math.isnan(temp_cost_m):
                print('Nan Value\n')
                return
            
            temp_cost += temp_cost_m / int(data/Nb)
        
        print(f"Epoch {epoch+1}: temp_cost = {temp_cost}")
        LOSS_Total.append(temp_cost)

        # Check if the current loss is the minimum
        if temp_cost < min_loss and epoch > 1900:
            print(f"New best model found at epoch {epoch+1}, saving model.")
            min_loss = temp_cost
            
            t1_save = time.time()

            torch.save(model.state_dict(), best_model_path)

            t2_save = time.time()
            t_save += t2_save - t1_save
        
        if temp_cost < J_Loss:
            break 
        
    end_ite = time.time()       
    information.close()    
    plotCost(LOSS_Total,'Loss','Loss')

    with open('track.txt', 'w') as file:
        file.writelines(f"{item}\n" for item in LOSS_Total)

    # Load the best model
    print("Loading the best model from saved checkpoint...")
    model.load_state_dict(torch.load(best_model_path))

    X_prediction = torch.tensor(X_train, dtype=model.parameters().__next__().dtype)
    X_prediction = X_prediction.permute(0, 2, 1).to(device)

    with torch.no_grad(): 
        prediction = model(X_prediction)

    prediction = prediction.cpu().numpy()

    for index in range(data):
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],CFD_train[index,:,0],index,'u truth '+str(index),r'Ground truth of velocity $u$ (m/s)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],prediction[index,0,:],index,'u prediction '+str(index),r'Prediction of velocity $u$ (m/s)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],CFD_train[index,:,1],index,'v truth '+str(index),r'Ground truth of velocity $v$ (m/s)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],prediction[index,1,:],index,'v prediction '+str(index),r'Prediction of velocity $v$ (m/s)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],CFD_train[index,:,2],index,'p truth '+str(index),r'Ground truth of pressure $p$ (Pa)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],prediction[index,2,:],index,'p prediction '+str(index),r'Prediction of pressure $p$ (Pa)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],CFD_train[index,:,3],index,'T truth '+str(index),r'Ground truth of temperature $T$ (K)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],prediction[index,3,:],index,'T prediction '+str(index),r'Prediction of temperature $T$ (K)')
            
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],np.abs(CFD_train[index,:,0]-prediction[index,0,:]),index,'abs error u'+str(index),r'Absolute error of velocity $u$ (m/s)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],np.abs(CFD_train[index,:,1]-prediction[index,1,:]),index,'abs error v'+str(index),r'Absolute error of velocity $v$ (m/s)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],np.abs(CFD_train[index,:,2]-prediction[index,2,:]),index,'abs error p'+str(index),r'Absolute error of pressure $p$ (Pa)')
        plotSolutions2DPoint(X_train[index,:,0],X_train[index,:,1],np.abs(CFD_train[index,:,3]-prediction[index,3,:]),index,'abs error T'+str(index),r'Absolute error of temperature $T$ (K)')

    #Error Analysis
    error_u_rel = [] 
    error_v_rel = [] 
    error_p_rel = [] 
    error_T_rel = []     
    error_T_sur_rel = [] 

    for index in range(data):        
        error_u_rel.append(relative_l2_norm(CFD_train[index,:,0],prediction[index,0,:]))
        error_v_rel.append(relative_l2_norm(CFD_train[index,:,1],prediction[index,1,:]))
        error_p_rel.append(relative_l2_norm(CFD_train[index,:,2],prediction[index,2,:]))
        error_T_rel.append(relative_l2_norm(CFD_train[index,:,3],prediction[index,3,:]))    
        error_T_sur_rel.append(computeRelativeL2OnSurface(X_train,prediction[index,3,:],index))        

        print('\n')

    for index in range(data):
        print('data: ',index)
        print('error_u_rel:')
        print(error_u_rel[index])
        print('error_v_rel:')
        print(error_v_rel[index])
        print('error_p_rel:')
        print(error_p_rel[index])
        print('error_T_rel:')
        print(error_T_rel[index])
        print('error_T_surface_rel:')
        print(error_T_sur_rel[index])                                                
        print('\n') 

    print('max relative u:')
    print(max(error_u_rel))
    print(error_u_rel.index(max(error_u_rel)))
    print('min relative u:')
    print(min(error_u_rel))
    print(error_u_rel.index(min(error_u_rel)))

    print('\n')

    print('max relative v:')
    print(max(error_v_rel))
    print(error_v_rel.index(max(error_v_rel)))
    print('min relative v:')
    print(min(error_v_rel))
    print(error_v_rel.index(min(error_v_rel)))

    print('\n')

    print('max relative p:')
    print(max(error_p_rel))
    print(error_p_rel.index(max(error_p_rel)))
    print('min relative p:')
    print(min(error_p_rel))
    print(error_p_rel.index(min(error_p_rel)))

    print('\n')

    print('max relative T:')
    print(max(error_T_rel))
    print(error_T_rel.index(max(error_T_rel)))
    print('min relative T:')
    print(min(error_T_rel))
    print(error_T_rel.index(min(error_T_rel)))
        
    print('\n')

    print('max relative T surface:')
    print(max(error_T_sur_rel))
    print(error_T_sur_rel.index(max(error_T_sur_rel)))
    print('min relative T surface:')
    print(min(error_T_sur_rel))
    print(error_T_sur_rel.index(min(error_T_sur_rel)))
        
    print('\n')

    print('average relative u:')
    print(sum(error_u_rel)/len(error_u_rel))

    print('\n')

    print('average relative v:')
    print(sum(error_v_rel)/len(error_v_rel))

    print('\n')

    print('average relative p:')
    print(sum(error_p_rel)/len(error_p_rel))
        
    print('\n')

    print('average relative T:')
    print(sum(error_T_rel)/len(error_T_rel))
        
    print('\n')

    print('average relative T surface:')
    print(sum(error_T_sur_rel)/len(error_T_sur_rel))
        
    print('\n')

    print('training time (second):')
    print(end_ite - start_ite - t_save)
        
    print('\n')

    print('min loss:')
    print(min(LOSS_Total)) 

    print('\n')

    ############################

    j = error_u_rel.index(max(error_u_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,0]-prediction[j,0,:]),j,'max error rel u'+str(j),r'Absolute error of velocity $u$ (m/s)')

    j = error_u_rel.index(min(error_u_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,0]-prediction[j,0,:]),j,'min error rel u'+str(j),r'Absolute error of velocity $u$ (m/s)')
        
    j = error_v_rel.index(max(error_v_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,1]-prediction[j,1,:]),j,'max error rel v'+str(j),r'Absolute error of velocity $v$ (m/s)')
        
    j = error_v_rel.index(min(error_v_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,1]-prediction[j,1,:]),j,'min error rel v'+str(j),r'Absolute error of velocity $v$ (m/s)')
        
    j = error_p_rel.index(max(error_p_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,2]-prediction[j,2,:]),j,'max error rel p'+str(j),r'Absolute error of pressure $p$ (Pa)')
        
    j = error_p_rel.index(min(error_p_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,2]-prediction[j,2,:]),j,'min error rel p'+str(j),r'Absolute error of pressure $p$ (Pa)')
        
    j = error_T_rel.index(max(error_T_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,3]-prediction[j,3,:]),j,'max error rel T'+str(j),r'Absolute error of temperature $T$ (K)')
        
    j = error_T_rel.index(min(error_T_rel))
    plotSolutions2DPoint(X_train[j,:,0],X_train[j,:,1],np.abs(CFD_train[j,:,3]-prediction[j,3,:]),j,'min error rel T'+str(j),r'Absolute error of temperature $T$ (K)')
        
    for index in range(data):   
        compute_T_surface(X_train,prediction[index,3,:],index) 

###### Function call ######
build_model_Thermal()
