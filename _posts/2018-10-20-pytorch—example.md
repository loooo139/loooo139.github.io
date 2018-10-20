---
layout:     post
title:      pytorch——example
subtitle:   
date:       2018-10-20
author:     Frank
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - pytorch
    - deep learning
---


## numpy two simple nerual network


```python
# -*- coding: utf-8 -*-
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(50):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

    0 28345763.30101315
    1 23772402.32817691
    2 22776520.449052237
    3 22044406.952793974
    4 19909947.770016104
    5 16009857.610662105
    6 11442535.830983132
    7 7429892.834058248
    8 4607322.075971094
    9 2858185.9464960275
    10 1842890.8971145307
    11 1261060.8098037245
    12 919624.3268211978
    13 708680.907271432
    14 569788.1109480723
    15 472103.03219111427
    16 399256.4482770234
    17 342387.9193104921
    18 296477.42894312297
    19 258501.2305698286
    20 226650.30989516538
    21 199566.0577962181
    22 176324.18540431885
    23 156287.242178826
    24 138918.80958871686
    25 123800.61205085581
    26 110606.10971327398
    27 99045.33318774498
    28 88882.81905844546
    29 79921.48414568664
    30 71999.01440654718
    31 64981.123596740734
    32 58749.551152608896
    33 53198.92853162432
    34 48252.07917950647
    35 43832.20960826811
    36 39871.701733552145
    37 36316.98705262467
    38 33120.46639261523
    39 30239.735238671543
    40 27640.946238029734
    41 25293.154607263936
    42 23168.733790563707
    43 21244.09523613098
    44 19498.772160036096
    45 17912.82223621226
    46 16470.619670862812
    47 15157.440922867536
    48 13959.917766549586
    49 12866.886096326078



```python
import torch

```


```python
dtype=torch.float
device=torch.device('cuda')
N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in,device=device,dtype=dtype)
y=torch.randn(N,D_out,device=device,dtype=dtype)
w1=torch.randn(D_in,H,device=device,dtype=dtype)
w2=torch.randn(H,D_out,device=device,dtype=dtype)
```


```python
learning_rate=1e-6
for t in range(50):
    h=x.mm(w1)
    h_relu=h.clamp(min=0)
    y_pred=h_relu.mm(w2)
    loss=(y_pred-y).pow(2).sum().item()
    print(t,loss)
    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.t().mm(grad_y_pred)
    grad_h_relu=grad_y_pred.mm(w2.t())
    grad_h=grad_h_relu.clone()
    grad_h[h<0]=0
    grad_w1=x.t().mm(grad_h)
    w1-=learning_rate*grad_w1
    w2-=learning_rate*grad_w2
    
```

    0 33874024.0
    1 38042684.0
    2 50846984.0
    3 60654044.0
    4 51404972.0
    5 26705972.0
    6 9110502.0
    7 3147570.0
    8 1637065.5
    9 1172994.0
    10 945688.9375
    11 789969.4375
    12 669026.9375
    13 571425.375
    14 491209.3125
    15 424591.28125
    16 368864.875
    17 321902.71875
    18 282060.4375
    19 248061.25
    20 218903.59375
    21 193804.09375
    22 172083.390625
    23 153217.34375
    24 136778.5
    25 122378.765625
    26 109739.90625
    27 98618.0234375
    28 88796.046875
    29 80111.515625
    30 72417.234375
    31 65592.9765625
    32 59511.66015625
    33 54077.13671875
    34 49210.4375
    35 44843.140625
    36 40921.16015625
    37 37391.015625
    38 34208.1484375
    39 31333.869140625
    40 28734.9453125
    41 26382.28125
    42 24248.61328125
    43 22313.3125
    44 20555.005859375
    45 18952.6796875
    46 17491.64453125
    47 16157.5380859375
    48 14938.6591796875
    49 13824.982421875


## use autograd of pytorch to create nerual network


```python
import torch

dtype = torch.double
device = torch.device("cuda")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(50):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```

    0 27329798.15195738
    1 20284511.432770804
    2 18184038.824153844
    3 17845757.258747093
    4 17612284.454891887
    5 16325457.613050558
    6 13804425.056262748
    7 10496459.644218748
    8 7345048.242608876
    9 4847811.810430426
    10 3146274.893020187
    11 2060491.9772843712
    12 1395489.3610292312
    13 984839.3968740824
    14 726632.9514072824
    15 558123.9194926216
    16 443575.79747538373
    17 362088.9705187212
    18 301751.99945664767
    19 255478.45266043904
    20 218961.65663170602
    21 189467.53593040784
    22 165150.3906637196
    23 144805.57522778912
    24 127583.95431848403
    25 112882.14408143431
    26 100224.5720389265
    27 89253.34318989089
    28 79714.42167730333
    29 71359.74691416133
    30 64014.968080630344
    31 57536.28900010305
    32 51800.840093437335
    33 46715.42520373034
    34 42200.19610322229
    35 38172.84707592711
    36 34574.402943578236
    37 31353.875911157076
    38 28464.60506358549
    39 25871.676714559282
    40 23540.731132641213
    41 21441.96539832774
    42 19547.987216637717
    43 17836.173780881778
    44 16287.28942084763
    45 14884.603529693135
    46 13613.388706478307
    47 12459.527065124037
    48 11411.48559158315
    49 10458.328562491959



```python
torch.device('cuda')
```




    device(type='cuda')




```python
# -*- coding: utf-8 -*-
import torch


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(50):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```

    0 27589096.0
    1 24716268.0
    2 25901828.0
    3 27106104.0
    4 25592950.0
    5 20293266.0
    6 13550911.0
    7 7872054.0
    8 4351147.5
    9 2470048.0
    10 1530027.125
    11 1047923.625
    12 782785.9375
    13 621167.0625
    14 512024.90625
    15 431796.90625
    16 369219.5
    17 318535.28125
    18 276583.53125
    19 241311.46875
    20 211354.75
    21 185717.671875
    22 163660.03125
    23 144615.0625
    24 128121.390625
    25 113783.34375
    26 101261.3203125
    27 90340.0
    28 80762.046875
    29 72336.546875
    30 64901.80078125
    31 58321.13671875
    32 52484.81640625
    33 47298.546875
    34 42683.703125
    35 38567.140625
    36 34889.9375
    37 31600.322265625
    38 28651.751953125
    39 26010.830078125
    40 23638.63671875
    41 21504.869140625
    42 19585.068359375
    43 17852.984375
    44 16288.341796875
    45 14873.478515625
    46 13592.5751953125
    47 12433.021484375
    48 11380.30859375
    49 10424.5966796875


# pytorch:NN


```python
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(50):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
```

    0 696.6943359375
    1 648.1907958984375
    2 605.9769287109375
    3 568.757080078125
    4 535.3941040039062
    5 504.9647521972656
    6 477.0088806152344
    7 451.2073974609375
    8 427.18585205078125
    9 404.9003601074219
    10 384.0518493652344
    11 364.4468994140625
    12 345.8046569824219
    13 328.0224609375
    14 311.031982421875
    15 294.7994689941406
    16 279.29693603515625
    17 264.43414306640625
    18 250.2398223876953
    19 236.69935607910156
    20 223.770751953125
    21 211.401611328125
    22 199.5756378173828
    23 188.27809143066406
    24 177.49864196777344
    25 167.24375915527344
    26 157.48248291015625
    27 148.2084197998047
    28 139.397705078125
    29 131.05642700195312
    30 123.13725280761719
    31 115.6380615234375
    32 108.56412506103516
    33 101.88695526123047
    34 95.59281921386719
    35 89.67591857910156
    36 84.11248779296875
    37 78.89147186279297
    38 73.99828338623047
    39 69.41324615478516
    40 65.11067962646484
    41 61.08793640136719
    42 57.33142852783203
    43 53.8188362121582
    44 50.52811050415039
    45 47.452491760253906
    46 44.575435638427734
    47 41.89399337768555
    48 39.3915901184082
    49 37.04938507080078


# pytorch:optim


```python
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(50):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
```

    0 720.6553344726562
    1 702.3820190429688
    2 684.6144409179688
    3 667.3594970703125
    4 650.572265625
    5 634.222900390625
    6 618.30810546875
    7 602.7865600585938
    8 587.658447265625
    9 572.93310546875
    10 558.5911254882812
    11 544.6459350585938
    12 531.0933227539062
    13 517.9594116210938
    14 505.4117126464844
    15 493.2707214355469
    16 481.48602294921875
    17 470.0767822265625
    18 458.9979553222656
    19 448.20159912109375
    20 437.6811218261719
    21 427.4390563964844
    22 417.4325866699219
    23 407.7120361328125
    24 398.2447509765625
    25 388.9586181640625
    26 379.85919189453125
    27 370.9375
    28 362.26025390625
    29 353.8121643066406
    30 345.5559997558594
    31 337.5265197753906
    32 329.7105712890625
    33 322.06256103515625
    34 314.58782958984375
    35 307.2544250488281
    36 300.049560546875
    37 292.9859619140625
    38 286.06988525390625
    39 279.2793273925781
    40 272.6336975097656
    41 266.1187744140625
    42 259.7496032714844
    43 253.53335571289062
    44 247.43617248535156
    45 241.45948791503906
    46 235.62181091308594
    47 229.8969268798828
    48 224.2857666015625
    49 218.8058319091797


# pytorch: custom nn modules


```python
class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        # we instantiate two nn.linear modules and assign them as member variables
        super(TwoLayerNet,self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)
    def forward(self,x):
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)
        return y_pred
    
```


```python
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
```


```python
# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)
```


```python
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
```


```python
for t in range(50):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    0 692.61572265625
    1 610.361328125
    2 542.890625
    3 486.61077880859375
    4 438.737060546875
    5 397.75714111328125
    6 362.39715576171875
    7 331.61138916015625
    8 304.4349670410156
    9 280.1077575683594
    10 258.47894287109375
    11 238.96295166015625
    12 221.18856811523438
    13 205.07058715820312
    14 190.3840789794922
    15 176.9864044189453
    16 164.56790161132812
    17 153.18023681640625
    18 142.64297485351562
    19 132.8334503173828
    20 123.79204559326172
    21 115.4233169555664
    22 107.67436218261719
    23 100.4552993774414
    24 93.76690673828125
    25 87.56297302246094
    26 81.7870864868164
    27 76.41168212890625
    28 71.40351104736328
    29 66.74405670166016
    30 62.40553665161133
    31 58.37321472167969
    32 54.62108612060547
    33 51.13092803955078
    34 47.87405776977539
    35 44.83446502685547
    36 41.999267578125
    37 39.35222625732422
    38 36.883846282958984
    39 34.56597900390625
    40 32.40399169921875
    41 30.389331817626953
    42 28.51129913330078
    43 26.759559631347656
    44 25.120725631713867
    45 23.591794967651367
    46 22.162220001220703
    47 20.825241088867188
    48 19.574983596801758
    49 18.404739379882812


# dynamic network 


```python
# -*- coding: utf-8 -*-
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(50):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    0 645.5269165039062
    1 632.7625122070312
    2 668.4656372070312
    3 630.75
    4 616.5935668945312
    5 605.92138671875
    6 627.26806640625
    7 627.0993041992188
    8 572.1802978515625
    9 470.8485107421875
    10 550.5152587890625
    11 621.1929931640625
    12 526.7205200195312
    13 511.923828125
    14 623.8075561523438
    15 477.2958679199219
    16 334.38330078125
    17 621.6140747070312
    18 281.7661437988281
    19 618.4879150390625
    20 591.3341674804688
    21 583.3375854492188
    22 607.5723876953125
    23 155.8756866455078
    24 335.9293518066406
    25 586.641845703125
    26 575.8047485351562
    27 499.807861328125
    28 542.6661987304688
    29 258.42047119140625
    30 496.83349609375
    31 112.61947631835938
    32 220.05714416503906
    33 372.2060241699219
    34 416.13787841796875
    35 391.2891540527344
    36 360.5303955078125
    37 164.7849884033203
    38 303.738525390625
    39 274.2149963378906
    40 146.01873779296875
    41 199.63955688476562
    42 179.23162841796875
    43 159.8955078125
    44 170.86990356445312
    45 193.20358276367188
    46 150.99905395507812
    47 128.3212127685547
    48 137.1464080810547
    49 107.7003173828125