import torch 
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F


#from model import FNet
import torchvision
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def create_meta_csv(dataset_path, destination_path):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """

    # Change dataset path accordingly
    DATASET_PATH = os.path.abspath(dataset_path)
    print(DATASET_PATH)
    
    

    if not os.path.exists(os.path.join(DATASET_PATH, "/dataset_attr.csv")):

        # Make a csv with full file path and labels
        labels=os.listdir(DATASET_PATH+"/")
        print(labels)
        label=[]
        value=[]
        l_idx=[]
        k=0
        for i in labels:
            if i=='dataset_attr.csv' :
                continue
            for j in os.listdir(DATASET_PATH+"/"+i+"/"):
                l_idx.append(k)
                label.append([i,k])
                value.append(DATASET_PATH+"/"+i+"/"+j)
            k=k+1
        print(np.unique(l_idx))    
                
        # change destination_path to DATASET_PATH if destination_path is None 
        if destination_path == None:
            destination_path = dataset_path

        # write out as dataset_attr.csv in destination_path directory
        df=pd.DataFrame({'path':value,'label':label,'label_idx':l_idx})
        df.to_csv(destination_path+"/"+"dataset_attr.csv")

        # if no error
        return True

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a 
    fraction in split parameter.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    if create_meta_csv(dataset_path, destination_path=destination_path):
        dframe = pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'))

    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        # shuffle the dataframe here
        dframe=dframe.sample(frac=1)
        #pass

    if split != None:
        train_set, test_set = train_test_split(dframe, split)
        return dframe, train_set, test_set 
    
    return dframe

def train_test_split(dframe, split_ratio):
    """Splits the dataframe into train and test subset dataframes.

    Args:
        split_ration (float): Divides dframe into two splits.

    Returns:
        train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
        test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
    """
    # divide into train and test dataframes
    split_train=int(split_ratio*len(dframe))
    train_data=dframe[:split_train]
    test_data=dframe[split_train:]
    return train_data, test_data

class ImageDataset(Dataset):
    """Image Dataset that works with images
    
    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.
    
    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """

    def __init__(self, data, transform=None):
        
        self.data = data
        self.transform = transform
        self.classes = data['label'].unique()# get unique classes fr4  cgjom data dataframe
        self.label_idx=data['label_idx']
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        image = Image.open(img_path)# load PIL image
        label = self.data.iloc[idx]['label']# get label (derived from self.classes; type: int/long) of image
        label_idx=self.data.iloc[idx]['label_idx']
        if self.transform:
            image = self.transform(image)
            
        return image, label,label_idx



class FNet(nn.Module):
    # Extra TODO: Comment the code with docstrings
    """Fruit Net
    
    """

    def __init__(self):
        super(FNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(484416, 64)
        self.fc2 = nn.Linear(64, 5)
        
        #pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
       
        return out




def train_model(dataset_path, debug=False, destination_path='', save=False,epochs=1):
	"""Trains model with set hyper-parameters and provide an option to save the model.

	This function should contain necessary logic to load fruits dataset and train a CNN model on it. It should accept dataset_path which will be path to the dataset directory. You should also specify an option to save the trained model with all parameters. If debug option is specified, it'll print loss and accuracy for all iterations. Returns loss and accuracy for both train and validation sets.

	Args:
		dataset_path (str): Path to the dataset folder. For example, '../Data/fruits/'.
		debug (bool, optional): Prints train, validation loss and accuracy for every iteration. Defaults to False.
		destination_path (str, optional): Destination to save the model file. Defaults to ''.
		save (bool, optional): Saves model if True. Defaults to False.

	Returns:
		loss (torch.tensor): Train loss and validation loss.
		accuracy (torch.tensor): Train accuracy and validation accuracy.
	"""
	if os.path.isdir('./dataset_attr') == False:
		os.mkdir('./dataset_attr')
	dframe,train_data,test_data=create_and_load_meta_csv_df(dataset_path,destination_path,randomize=True,split=.9) #when you are testing or running the code you must have to create the folder dataset manualy it can be implemented using os.makedir but we were having issue , so we ave decided to create it manualy
	train_data=DataLoader(dataset=ImageDataset(data=train_data,transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
        ])),batch_size=16)
	test_data=DataLoader(dataset=ImageDataset(test_data,transforms.ToTensor()),batch_size=16)
	costFunction=nn.CrossEntropyLoss()
	optimizer=optim.SGD(net.parameters(),lr=.001,momentum=0.9)
	#print(train_data.dataset.data)
	acc,loss=train(epochs,train_data=train_data,costFunction=costFunction,optimizer=optimizer,test_data=test_data,debug=debug,destination_path=destination_path,save=save)

	return acc,loss



	# Write your code here
	# The code must follow a similar structure
	# NOTE: Make sure you use torch.device() to use GPU if available
	
def train(epochs,train_data,costFunction,optimizer,test_data,debug,destination_path,save):
	
	net.train().to(device)
	for epoch in range(epochs):
		losses=[]
		closs=5
		for i,(image,_,idx) in enumerate(train_data,0):
			pred=net(image)
			loss=costFunction(pred,idx)
			closs+=loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if debug == True :
				losses.append(loss.item())
				print('[%d %d] loss: %.6f'%(epoch+1,i+1,closs*10))
				closs=0
		acc=accuracy(test_data)
		plt.plot(losses,label='epoch'+str(epoch))
		plt.legend(loc=1,mode='expanded')
	plt.show()
	torch.save(net.state_dict(),'mode.pth')
	return acc,loss.item()/100	
	
	

def accuracy(test_data):
    net.eval().to(device)
    correcthit=0
    total=0
    accuracy=0
    for batches in test_data:
        data,label,idx=batches
        prediction=net(data)
        _,prediction=torch.max(prediction.data,1)
        total +=idx.size(0)
        correcthit +=(prediction==idx).sum().item()
        accuracy=(correcthit/total)*100
    print("accuracy = "+str(accuracy))
    return accuracy
           
                
                


if __name__ == "__main__":
    # test config
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    dataset_path = './dataset_bgr_rmv/'
    dest = './dataset_attr_rmv/'
    if os.path.isdir(dest) == False:
        os.mkdir(dest)
    classes = 5
    #total_rows = 304
    randomize = True
    clear = True
    
    # test_create_meta_csv()

    df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path,destination_path=dest, randomize=randomize, split=0.9)
    print(df.describe())
    print(trn_df.describe())
    print(tst_df.describe())
    
    net=FNet().to(device)	
    print(net)
    for parameter in net.parameters():
        print(str(parameter.data.numpy().shape)+'\n')

    acc,loss=train_model(dataset_path='./dataset_bgr_rmv/',destination_path='./data_attr_rmv/',epochs=17,debug=True,save=True)
    print("accuracy :",acc)
    print('loss : %.4f'%loss)