function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

e_min = inf;
values = [0.01 0.03 0.1 0.3 1 3 10 30];
for tmp_C = values
  for tmp_sigma = values
    model = svmTrain(X, y, tmp_C, @(x1, x2) gaussianKernel(x1, x2, tmp_sigma));
    err = mean(double(svmPredict(model, Xval) ~= yval));
    if (err <= e_min)
      C = tmp_C;
      sigma = tmp_sigma;
      e_min = err;
      fprintf('updated C, sigma, e_min = %f, %f, %f\n', C, sigma, e_min); 
    end
  end
end

fprintf('the best parameters C, sigma = %f, %f\n', C, sigma);


% =========================================================================

end
