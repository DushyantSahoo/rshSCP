function [gradients,state,grad_input_mse,loss_cross,accuracy] = modelGradients_feature_predict(dlnet,dlW,Y,k)

scaleFactor = ones(k,1);
offset = zeros(k,1);
[dlYPred1,state] = forward(dlnet,batchnorm(dlW,offset,scaleFactor));
loss_cross = crossentropy(dlYPred1,Y);
gradients = dlgradient(loss_cross,dlnet.Learnables);
grad_input_mse = dlgradient(loss_cross,dlW);
accuracy = sum(onehotdecode(double(extractdata(Y)),[1,2],1)==onehotdecode(double(extractdata(dlYPred1)),[1,2],1))/length(Y);
%grad_input_mse = grad_input_mse(1:(end-1),:);
end