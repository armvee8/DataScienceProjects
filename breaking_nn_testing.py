import numpy as np 

#HEAVY WIP, just for fun + learning

epsilons = [0, 0.25, 0.50, 0.75]
pretrained_model = 'filepath' #place filepath of the model here 
use_cuda = True 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #constructor
        self.conv1 = nn.Conv2d(1,10, 5)
        self.conv2 = nn.Conv2d(10,20, 5)
        self.conv2_drop = nn.Dropout2d() 
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10) 
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform = transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True)

#TODO: IMPLEMENT CUDA DEVICE
device = 'temp'

#Initialize the network
model = Net().to(device)

#Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

#Set the model in evaluation mode.
model.eval() 

def fgsm_attack(image, epsilon, data_grad):
    #Collect the element-wise sign of the data gradient
    sign_data-grad = data_grad.sign() 
    #Create the perturbed iamge by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    #Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image 

def test(model, device, test_loader, epsilon): 
    #Accuracy counter
    correct = 0
    adv_examples = [] 

    #Loop over all examples in test set
    for data, target in test_loader:
        #Send the data and label to the device 
        data, target = data.to(device), target.to(device)
        #Set requires_grad attribute of tensor, important for attack
        data.requires_grad = True
        #Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] #grabs index of the max logistic probability 
        
