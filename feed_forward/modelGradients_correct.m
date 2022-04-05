function [gradients,state,grad_input_cross,loss_sex,loss_cross,loss_mse] = modelGradients_correct(dlnet,dlX,Y_label,Y_age,Y_sex,k,correct)

scaleFactor = ones(k,1);
offset = zeros(k,1);
%[dlYPred_label,dlYPred_age,dlYPred_sex,state] = forward(dlnet,dlX);

[dlYPred_label,dlYPred_age,dlYPred_sex,state] = forward(dlnet,batchnorm(dlX,offset,scaleFactor));

loss_cross = correct*crossentropy(dlYPred_label,Y_label);
loss_sex = crossentropy(dlYPred_sex,Y_sex);
temp = (dlYPred_age-Y_age).^2;
loss_mse = sqrt(mean(((temp))));
total_loss = loss_cross+loss_mse+loss_sex;
%total_loss = loss_cross*(loss_mse+loss_sex)/total_loss1 ...
%            +loss_mse*(loss_cross+loss_sex)/total_loss1...
%            +loss_sex*(loss_cross+loss_mse)/total_loss1;
gradients = dlgradient(total_loss,dlnet.Learnables);
grad_input_cross = dlgradient(total_loss,dlX);
%loss = double(gather(extractdata(loss)));

end