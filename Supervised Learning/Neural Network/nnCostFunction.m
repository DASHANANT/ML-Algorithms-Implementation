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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
z1 = [ones(m, 1) X] * Theta1';
a1 = sigmoid(z1);
z2 = [ones(m, 1) a1] * Theta2';
a2 = sigmoid(z2);
% J = (-1/m)* ((y*log(h')) + ((1-y)*log(1-h'))); my first implemtation
% Note the values of the h and y are the clases 1,2,3 and for our cost
% function to work this must be 1
v = eye(m,num_labels);
v = v(y,:);
initJ = (v.*log(a2)) + ((1-v).*log(1-a2)); % Question why this is not working (v'*log(h)) + ((1-v)'*log(1-h))
J_unreg = (-1/m)* sum(initJ(:)); % Unregularized cost function
theta1_sq = Theta1(:,2:end).^2;
theta2_sq = Theta2(:,2:end).^2;
reg = lambda/(2*m) * (sum(theta1_sq(:)) + sum(theta2_sq(:))); % Regularization
J = J_unreg + reg;

error3 = a2 - v; % a2 - y this is what I was using before and I am having error
error2 = (error3*Theta2(:,2:end)).*sigmoidGradient(z1);

Theta1_grad = (1/m)*(error2'*[ones(m, 1) X] + lambda * [zeros(hidden_layer_size, 1) Theta1(:, 2:end)]);
Theta2_grad = (1/m)*(error3'*[ones(m, 1) a1] + lambda * [zeros(num_labels, 1) Theta2(:, 2:end)]);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
