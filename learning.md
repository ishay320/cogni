# neural network

## learning cycle

1. Taking the input data
2. Making a prediction
3. Comparing the prediction to the desired output
4. Adjusting its internal state to predict correctly the next time

## forward pass

1. every layer is liner layer means the its looks like

   $$price = (weight_{area} * area) + (weight_{age} * age) + bias$$
   `or`
   $$price = weight_a*area^2 + weight_b*area + bias$$

2. in between layers there is non linear function like `sigmoid` but it needs to able to derive

## backward pass (teaching the network) using gradient descent

1. calculate error function like `mse` that gets the prediction and the ground truth and output the calculated error
2. from the error function derive each layer back and apply the derivative to the parameters

### example:

the neural network

```mermaid
stateDiagram-v2
    x1 --> h1: w1
    x2 --> h1: w2
    h1 --> o1: w5
    x2 --> h2: w4
    x1 --> h2: w3
    h2 --> o1: w6
    o1:o1 (+b3)
    h1:h1 (+b1)
    h2:h2 (+b2)
```

we have some function $sigmoid(x)$ that trying to predict y, we will use the `mse` function: $mse = (sigmoid(x) - y)^2$ <br/>
then we need to derive `mse` $mse' = 2(f(x) - y)$ <br/>
then
$$sigmoid(x) = \frac{1}{1+e^{-x}}$$
and the derivative is $sigmoid(x)' = sigmoid(x)*(1-sigmoid(x))$<br/>
then after derive sigmoid we need to derive forward, lets say that $x$ is $2a^3 + 4b^2 + c + 6$ then we derive for each var: <br/>
a: $6a^2$
b: $8b$
c: $1$

## forward pass

$$h_1(x_1,x_2) = x_2w_2 + x_1w_1 + b_1$$
$$h_2(x_1,x_2) = x_2w_3 + x_1w_4 + b_2$$
$$o_1(h_1,h_2) = h_2w_5 + h_1w_6 + b_3$$
$$mse(y_{pred}, y_{true}) = (y_{pred} - y_{true})^2$$

and on every layer there is an activision function:
$$f(x) = sigmoid(x) = \frac{1}{1+e^{-x}}$$

## chain rule

we have neural network as the graph shows<br/>
we can write the loss function using the weights:
$$L(w_1,w_2,w_3,w_4,w_5,w_6,b_1,b_2,b_3)$$
then if we wants to derive $w_1$ it will be $\frac{\partial L} {\partial w_1}$
then by the chain rule it can be

$$\frac{\partial L} {\partial w_1} = \frac{\partial L}{\partial y_{pred}}\cdot \frac{\partial y_{pred}}{\partial w_1}$$

we can calculate $\frac{\partial L}{\partial y_{pred}}$ because its just $mse$ and lets say that $y_{true} = 1$

$$L = (1-y_{pred}) \implies \frac{\partial L}{\partial y_{pred}} =L' = -2(1-y_{pred})$$

because $y_{pred} = o_1 = h_2w_5 + h_1w_6 + b_3$ and $w_1$ is in $h_1$ we can calculate

$$\frac{\partial y_{pred}}{\partial w_1} = \frac{\partial y_{pred}}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$$

then we derive for $h_1$
$$\frac{\partial y_{pred}}{\partial h_1} = \frac{\partial f(h_2w_5 + h_1w_6 + b_3)}{\partial h_1} = w_6f'(h_2w_5 + h_1w_6 + b_3)$$

then we derive the last layer

$$\frac{\partial h_1}{\partial w_1} = \frac{\partial h1(x_2w_2 + x_1w_1 + b_1)}{\partial w_1} = x_1f'(x_2w_2 + x_1w_1 + b_1)$$

and all we left is to derive the activision of each layer, the $f(x)$ function

$$
f(x) = (\frac{1}{1+e^{-x}})' = \frac{e^{-x}}{(1+e^{-x})^2} = f(x)\cdot(1-f(x))
$$

when we combine the
$$\frac{\partial L} {\partial w_1} = \frac{\partial L}{\partial y_{pred}}\cdot \frac{\partial y_{pred}}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$$

$$
\frac{\partial L} {\partial w_1} = (-2(1-y_{pred})) \cdot  w_6f'(h_2w_5 + h_1w_6 + b_3) \cdot x_1f'(x_2w_2 + x_1w_1 + b_1) \\
= (-2(1-y_{pred})) \cdot  \\
w_6(f(h_2w_5 + h_1w_6 + b_3)\cdot(1-f(h_2w_5 + h_1w_6 + b_3))) \cdot \\
 x_1(f(x_2w_2 + x_1w_1 + b_1)\cdot  (1-f(x_2w_2 + x_1w_1 + b_1)))
$$
