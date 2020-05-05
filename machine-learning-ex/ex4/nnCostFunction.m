function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%










% feedforward
% layer1
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% layer2
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Expand the 'y' output values into a matrix of single values
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);





% Caculate J
J = 1 / m * sum(sum(-y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3)));

% Toss the first columns of each Theta(i) matrix.
regularizedTheta1 = Theta1(:,2:size(Theta1,2));
regularizedTheta2 = Theta2(:,2:size(Theta2,2));
penalty =  sum(sum(regularizedTheta1.^2 )) + sum(sum(regularizedTheta2.^2 ));


J = J + lambda / 2 / m * penalty;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% -------------------------------------------------------------
% Backpropagation
for i = 1:m
    % Forward propagation
    x_t = X(i, :);
    x_t = [1, x_t]; % add bias 
    
    % ======= LAYER 2 =======
    z2_t = x_t * Theta1';
    a2_t = sigmoid(z2_t);% add bias to vector a2_t
    a2_t = [1 , a2_t];
    
    % ======= LAYER 3 =======
    % sigmoid input (can be vector or matrix - depending on Theta2)
    z3_t = a2_t * Theta2';
    
    % activation unit (this can be matrix or vector, depending on output of sigmoid)
    a3_t = sigmoid(z3_t);
    
    % compute output error
    y_t = y_matrix(i , :);
    delta_3 = (a3_t - y_t);
    
    % compute hidden layer error
    tmp = Theta2' * delta_3';
    
    % ignore the bias term and calculate error for layer 2
    delta_2 = tmp(2:end , :)' .* sigmoidGradient(z2_t);
    
    % Transition matrix error accumulators
    Theta1_grad = Theta1_grad + delta_2' * x_t;
    Theta2_grad = Theta2_grad + delta_3' * a2_t;
end
 
% Now divide everything (element-wise) by m to return the partial
% derivatives. Note that for regularization these will have to
% removed/commented out.

%Theta2_grad = Theta2_grad / m;
%Theta1_grad = Theta1_grad / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
