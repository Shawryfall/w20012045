% Load dataset
data = readtable('FitBit data.csv');

% Extract data for TotalSteps and TotalDistance
TotalSteps = data.TotalSteps;
TotalDistance = data.TotalDistance;

% Scale TotalSteps
TotalSteps = (TotalSteps - min(TotalSteps)) / (max(TotalSteps) - min(TotalSteps));

% Split the dataset into training (80%) and testing (20%) subsets
rng('default'); % For reproducibility
splitRatio = 0.8;
splitIndex = floor(height(data) * splitRatio);
randomIndices = randperm(height(data));
trainingIdx = randomIndices(1:splitIndex);
testingIdx = randomIndices(splitIndex+1:end);

% Create a feedforward neural network using the training subset
X_train = TotalSteps(trainingIdx);
y_train = TotalDistance(trainingIdx);

% Set up the neural network
net = feedforwardnet(10); % 1 hidden layer with 10 neurons
net = train(net, X_train', y_train');

% Test the neural network using the testing subset
X_test = TotalSteps(testingIdx);
y_test = TotalDistance(testingIdx);
y_pred_nn = net(X_test');

% Calculate and display the Mean Squared Error (MSE)
mse_nn = mean((y_test - y_pred_nn').^2);
fprintf('Neural Network regression(Min-Max Scaling) Mean Squared Error: %.2f\n', mse_nn);

% Calculate and display the Mean Absolute Error (MAE)
mae_nn = mean(abs(y_test - y_pred_nn'));
fprintf('Neural Network regression(Min-Max Scaling) Mean Absolute Error: %.2f\n', mae_nn);

% Calculate and display the Residual Standard Error (RSE)
n = length(y_test);
p = numel(net.layers) + sum(net.layers{1}.size); % Number of predictor variables (layers + neurons)
RSS = sum((y_test - y_pred_nn').^2);
RSE = sqrt(RSS / (n - p - 1));
fprintf('Neural Network regression(Min-Max Scaling) Residual Standard Error (RSE): %.2f\n', RSE);

% Predict TotalDistance for new TotalSteps
newTotalSteps = 10000; % Example value
newTotalSteps_scaled = (newTotalSteps - min(data.TotalSteps)) / (max(data.TotalSteps) - min(data.TotalSteps)); % Scale the new TotalSteps
predictedTotalDistance = net(newTotalSteps_scaled);

% Display the predicted TotalDistance
fprintf('Predicted TotalDistance for %d TotalSteps: %.2f\n', newTotalSteps, predictedTotalDistance);
