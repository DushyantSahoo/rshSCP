function [W, lambda, error,C,ld,dlnet_predict,dlnet_correct,accuracy_train] = rhscp_learn(A,k,alpha,loop,eta,beta1,beta2,eps,tole, index,gamma_predict,gamma_correct,features,age,sex)

correct = 1;
total_s = length(unique(index));
for i =1:max(index)
    ratio(i) = nnz(index==i)/nnz(index);
end
invert_r = 1./ratio;
invert_r = invert_r/total_s;

start_hetero = 1;
start_prediction = 400;
start_correction = 450;
correction = 0;
set(groot,'defaultLineLineWidth',6.0)
train_index = ones(length(A),1);
svd_check = 1;
plot_on = 1;
if plot_on ==1
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098],'LineWidth',2);
    lineLosstest = animatedline('Color',[0.25 0.325 0.098],'LineWidth',2);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Accuracy")
    grid on
end

hierarchy = length(k)-1;
% hierarchy - number of hierarchical components
% k contains the number of components in each hierarchy
inputSize = 0;
for hi=1:hierarchy
    inputSize = inputSize + k(hi+1);
end

% classification + regression
layers1 = [ ...
    featureInputLayer(inputSize,'Name','input')
    batchNormalizationLayer('Name','batch')
    fullyConnectedLayer(10,'Name','full')
    %dropoutLayer(0.4,'Name','dropout1')
    tanhLayer('Name','tanh')
    fullyConnectedLayer(10,'Name','full1')
    tanhLayer('Name','tanh1')
    %dropoutLayer(0.4,'Name','dropout2')
    fullyConnectedLayer(total_s,'Name','fc1')
    softmaxLayer('Name','softmax')];
lgraph_correct = layerGraph(layers1);
layers2 = [ ...
    fullyConnectedLayer(10,'Name','rfull12')
    %dropoutLayer(0.4,'Name','rdropout22')
    fullyConnectedLayer(1,'Name','fc22')];
lgraph_correct = addLayers(lgraph_correct,layers2);
layers3 = [ ...
    fullyConnectedLayer(10,'Name','rfull13')
    %dropoutLayer(0.4,'Name','rdropout23')
    fullyConnectedLayer(2,'Name','fc13')
    softmaxLayer('Name','softmax1')];
lgraph_correct = addLayers(lgraph_correct,layers3);
lgraph_correct = connectLayers(lgraph_correct,'tanh','rfull12');
lgraph_correct = connectLayers(lgraph_correct,'tanh','rfull13');

% learning from features
layers_predict = [ ...
    featureInputLayer(inputSize,'Name','input')
    dropoutLayer(0.5,'Name','dropout1')
    %fullyConnectedLayer(5,'Name','full')
    %dropoutLayer(0.4,'Name','dropout2')
    %tanhLayer('Name','tanh')
    %fullyConnectedLayer(20,'Name','full1')
    %dropoutLayer(0.4,'Name','dropout2')
    fullyConnectedLayer(2,'Name','fc1')
    softmaxLayer('Name','softmax')];

lgraph_predict = layerGraph(layers_predict);
dlnet_predict = dlnetwork(lgraph_predict);


dlnet_correct = dlnetwork(lgraph_correct);
averageGrad_predict = [];
averageSqGrad_predict = [];
averageGrad_correct = [];
averageSqGrad_correct = [];
averageGrad1 = [];
averageSqGrad1 = [];
averageGrad2 = [];
averageSqGrad2 = [];
initialLearnRate = 0.01;
decay = 0.01;

gradDecay = 0.9;
sqGradDecay = 0.99;
scanners = max(unique(index));
%labels = onehotencode(categorical(cellstr(num2str(index))),2);

[~,subjects] = size(A);


svd_error = 1;
for hi=1:hierarchy
    
    
    W{hi} = zeros(k(hi),k(hi+1));
    lambda_avg{hi} = zeros(k(hi+1),k(hi+1));
    for sub3=1:subjects
        
        if hi==1
            %U = = randn
            %A{sub3} = vineBeta(n,1);
            [U,S,V]=svd(A{sub3});
            W{hi} = W{hi}+ U(:,1:k(hi+1));
            lambda{hi,sub3} = S(1:k(hi+1),1:k(hi+1));
            lambda{hi,sub3} = diag(ones(k(hi+1),1));
            if svd_check==0
                lambda{hi,sub3} = normrnd(0,0.05,k(hi+1),k(hi+1));
                W{hi} = normrnd(0,0.05,size(W{hi}));
                %lambda{hi,sub3} = lambda_in{hi,sub3};
                %W{hi} = W_in{hi};
            end
            lambda_avg{hi} = lambda_avg{hi}  + lambda{hi,sub3};
            %svd_error = svd_error+ norm(A{sub3} - U(:,1:k(hi+1))*S(1:k(hi+1),1:k(hi+1))*V(1:k(hi+1),:),'fro')/norm(A{sub3},'fro');
        else
            [U1,S1,~]=svd(lambda{(hi-1),sub3});
            %[U,S,V]=svd(A{sub3});
            
            lambda{hi,sub3} = S1(1:k(hi+1),1:k(hi+1));
            lambda{hi,sub3} = diag(ones(k(hi+1),1));
            W{hi} = W{hi} + U1(:,1:k(hi+1));
            
            if svd_check==0
                %lambda{hi,sub3} = lambda_in{hi,sub3};
                %W{hi} = W_in{hi};
                lambda{hi,sub3} = normrnd(0,0.05,k(hi+1),k(hi+1));
                W{hi} = normrnd(0,0.05,size(W{hi}));
            end
            lambda_avg{hi} = lambda_avg{hi}  + lambda{hi,sub3};
            %svd_error = svd_error+ norm(A{sub3} - U(:,1:k(hi+1))*S(1:k(hi+1),1:k(hi+1))*V(1:k(hi+1),:),'fro')/norm(A{sub3},'fro');
            
        end
        past_m_lambda{hi,sub3} = zeros(size(lambda{hi,sub3}));
        past_v_lambda{hi,sub3} = zeros(size(lambda{hi,sub3}));
    end
    
    past_v_W{hi} = zeros(size(W{hi}));
    past_m_W{hi} = zeros(size(W{hi}));
    
    past_v_lambda_avg{hi} = zeros(size(lambda_avg{hi}));
    past_m_lambda_avg{hi} = zeros(size(lambda_avg{hi}));
    
    if svd_check==1
        W{hi} = W{hi}/subjects;
        
    end
    lambda_avg{hi} = zeros(size(lambda_avg{hi}));
    svd_error=svd_error/(subjects*hierarchy);
end
for hi=1:hierarchy
    for indi = 1:scanners
        C{indi,hi} = zeros(k(hi+1),k(hi+1));
        past_v_C{indi,hi} = zeros(k(hi+1),k(hi+1));
        past_m_C{indi,hi} = zeros(k(hi+1),k(hi+1));
    end
    
    for indi = 1:scanners
        ld{indi,hi} = diag(diag(ones(k(hi+1),k(hi+1))));
        past_v_ld{indi,hi} = zeros(k(hi+1),k(hi+1));
        past_m_ld{indi,hi} = zeros(k(hi+1),k(hi+1));
    end
    
    
    for sub3=1:subjects
        %C{index(sub3),hi} = C{index(sub3),hi} +  A{sub3} - W{1}*lambda{1,sub3}*W{1}' - W{1}*lambda_avg{1}*W{1}';
    end
    for indi = 1:scanners
        %C{indi,hi} = diag(diag(C{indi,hi}/nnz(index==indi)));
    end
    
end
total_data = zeros(subjects,inputSize);
start_tic = tic;

influe = 1;
labels = onehotencode(categorical(cellstr(num2str(index))),2);
sex_labels = onehotencode(categorical(cellstr(num2str(sex'))),2);
features_labels = onehotencode(categorical(cellstr(num2str(features'))),2);
kk = ones(length(train_index),1);
for l = 1:loop
    
    Y{1} = W{1};
    for hi=2:hierarchy
        Y{hi} = Y{hi-1}*W{hi};
    end
    error(l) = 0;
    cross_loss(l) = 0;
    for hi=1:hierarchy
        for i = 1:scanners
            grad_C{i,hi} = zeros(k(hi+1),k(hi+1));
            grad_ld{i,hi} = zeros(k(hi+1),k(hi+1));
            
        end
        grad_lambda_avg{hi} = zeros(k(hi+1),k(hi+1));
        
        
        for sub = 1:subjects
            if hi ==1
                total_data(sub,1:k(hi+1)) = diag(lambda{hi,sub});
            else
                total_data(sub,k(hi)+1:(k(hi)+k(hi+1))) = diag(lambda{hi,sub});
            end
            
            if l>start_hetero
                new_A{sub,hi} = A{sub};
                %R = sprandn(k(1),k(1),0.3);
                %R(R>0)=1;
                %R = full(R);
               % new_A{sub,hi}=new_A{sub,hi}.*R;
                new_A_avg{sub,hi} = A{sub}-Y{hi}*(ld{index(sub),hi}*lambda{hi,sub}+correction*(C{index(sub),hi}))*Y{hi}';
                error(l) = error(l) + (norm(A{sub}-Y{hi}*(ld{index(sub),hi}*lambda{hi,sub}+C{index(sub),hi})*Y{hi}','fro'))^2/norm(A{sub},'fro')^2;
                
            else
                new_A_avg{sub,hi} = A{sub};
                new_A{sub,hi} = A{sub};
                error(l) = error(l) + (1+influe)*(norm(A{sub}-Y{hi}*lambda_avg{hi}*Y{hi}','fro')/norm(A{sub},'fro'))^2;
                
            end
            
            for new = hi:hierarchy
                if hi-new==0
                    X{hi,new,sub} = ld{index(sub),hi}*lambda{hi,sub}+C{index(sub),hi};
                    X_avg{hi,new,sub} = lambda_avg{hi};
                else
                    temp = W{new};
                    if (new-hi-1 ~=0)
                        for temp11=1:new-hi-1
                            temp=W{new-temp11}*temp;
                        end
                    end
                    X{new,hi,sub}=  temp*(ld{index(sub),new}*lambda{new,sub}+C{index(sub),new})*temp';
                    X_avg{new,hi}=  temp*lambda_avg{new}*temp';
                end
            end
        end
    end
    
    dlX1 = dlarray((total_data)','CB');
    %distance_points= corr(total_data(train_index,:)',total_data(test_index,:)');
    %distance_points= pdist2(total_data(train_index,:),total_data(test_index,:));

    %[~,I] = max(distance_points);
    temp_features = features;
    temp_features_test = features;
    %for i =1:sum(test_index)
    %    [~,I1] = sort(distance_points(:,i),'ascend');
    %    mean_val(i)= mean(temp_features(I1(1:50)));
        
    %end
    %temp_features_test(test_index) = mean_val;
    temp_features_test(temp_features_test>0.5) = 1;
    temp_features_test(temp_features_test<=0.5) = 0;
    temp_features_test = onehotencode(categorical(cellstr(num2str(temp_features_test'))),2);
    %dlX_predict = dlarray((([total_data]))','CB');
    dlY_predict = dlarray((features_labels)','CB');
    temp_dlY_predict = dlarray((temp_features_test)','CB');
    clear temp_features_test
    dlY_predict_test = dlarray((features_labels)','CB');
    dlY_labels= dlarray((labels)','CB');
    age = zscore(age);
    dlY_age= dlarray((age),'CB');
    dlY_sex= dlarray((sex_labels)','CB');
    loss_sex = dlarray(0);
    loss_predict = dlarray(0);
    loss_mse = dlarray(0);
    loss_correct = dlarray(0);
    accuracy_train = 0;
    accuracy_test = 0;
    dlX2 = dlarray(dlX1,'CB');
    if l > start_prediction
        [gradients,state,grad_input_correct,loss_sex,loss_correct,loss_mse] = dlfeval(@modelGradients_correct,dlnet_correct,dlX1,dlY_labels,dlY_age,dlY_sex,inputSize,correct);
        dlnet_correct.State = state;
        [dlnet_correct,averageGrad_correct,averageSqGrad_correct] = adamupdate(dlnet_correct,gradients,averageGrad_correct,averageSqGrad_correct,l-start_prediction,initialLearnRate,gradDecay,sqGradDecay);
        if l > 80
            
            
            if gamma_correct > 0
                [dlX1,averageGrad2,averageSqGrad2] = adamupdate(dlX1,-grad_input_correct,averageGrad2,averageSqGrad2,l-80,gamma_correct*initialLearnRate,gradDecay,sqGradDecay);
            end
            %end
            
        end
        dlX2 = dlarray(dlX1,'CB');
        if gamma_predict>0
            
            [gradients,state,grad_input_predict,loss_predict,accuracy_train] = dlfeval(@modelGradients_feature_predict,dlnet_predict,dlX1,temp_dlY_predict,inputSize);
            dlnet_predict.State = state;
            [dlnet_predict,averageGrad_predict,averageSqGrad_predict] = adamupdate(dlnet_predict,gradients,averageGrad_predict,averageSqGrad_predict,l-start_prediction,initialLearnRate,gradDecay,sqGradDecay);
            %final_grad_input_predict(:,test_index) = grad_input_predict(:,I);
            %final_grad_input_predict(:,train_index) = grad_input_predict;
            
            if l > start_correction
                
                
                [dlX1,averageGrad1,averageSqGrad1] = adamupdate(dlX1,grad_input_predict,averageGrad1,averageSqGrad1,l-start_correction,gamma_predict*initialLearnRate,gradDecay,sqGradDecay);
            end
            scaleFactor = ones(inputSize,1);
            offset = zeros(inputSize,1);
            [dlYPred1] = predict(dlnet_predict,batchnorm(dlarray(dlX1,'CB'),offset,scaleFactor));
            true_values = double(onehotdecode(double(extractdata(dlY_predict_test)),[1,2],1));
            pred_test_values = double(onehotdecode(double(extractdata(dlYPred1)),[1,2],1));
            check = pred_test_values==true_values;
            accuracy_test = sum(check(true_values==2))/sum(true_values==2) + sum(check(true_values==1))/sum(true_values==1);
            
            accuracy_test = accuracy_test/2;
        end
        
    end
    
    
    
    total_data = double(extractdata(dlX1));
    total_data = total_data';
    %total_data2 = double(extractdata(dlX2));
    %total_data2 = total_data2';
    %total_data(train_index,:) = total_data2;
    % Display the training progress.
    error(l)=error(l)/(hierarchy*subjects*(1+influe));
    if plot_on ==1
        
        D1 = duration(0,0,toc(start_tic),'Format','hh:mm:ss');
        %addpoints(lineLossTrain,l,double(extractdata(loss_predict)))
        %addpoints(lineLossTrain,l,error(l))
        addpoints(lineLossTrain,l,accuracy_train)
        addpoints(lineLosstest,l,accuracy_test)
        
        %title("Epoch: " + l + ", Elapsed: " + string(D1))
        drawnow
        set(gca,'FontSize',18)
        legend('Training Accuracy','Test Accuracy')
    end
    
    
    
    if (l >800)
        %temp_tol = (error(l-1000)-error(l))/error(l-1000);
        %temp_tol1 = (error(l-100)-error(l))/error(l-100);
        temp_tol2 = (error(l-100)-error(l))/error(l-100);
        %if((abs(temp_tol)<tole) && (abs(temp_tol1)<tole) && (abs(temp_tol2)<tole))
        %if( (abs(temp_tol1)<tole) && (abs(temp_tol2)<tole))
        if( (abs(temp_tol2)<tole))
            return
        end
        
    end
    for hi=1:hierarchy
        
        %cluster(1,:)= mean(total_data(features(train_index)==1,:));
        %cluster(2,:)= mean(total_data(features(train_index)==0,:));
        
        for sub = 1:subjects
            if l > start_hetero
                if correct == 1
                    if l > start_correction
                        correction = 1;
                    end
                end
                
                if hi ==1
                    lambda{hi,sub} = diag(total_data(sub,1:k(hi+1)));
                else
                    lambda{hi,sub} = diag(total_data(sub,k(hi)+1:(k(hi)+k(hi+1))));
                end
                %temp_clust2 = 200;
                
                %if rem(l,300)==0
                %    nnz_val = find(train_index);
                %    for trav = 1:sum(train_index)
                %        sub1 = nnz_val(trav);
                %        
                %        temp_clust1 = (norm(A{sub}-Y{hi}*lambda_avg{hi}*Y{hi}'-Y{hi}*(ld{index(sub),hi}*lambda{hi,sub1}+C{index(sub),hi})*Y{hi}','fro'))^2;
                %        if temp_clust2 > temp_clust1
                %            kk(sub) = sub1;
                %            temp_clust2 = temp_clust1;
                %        end
                %    end
                %    lambda{hi,sub}=lambda{hi,kk(sub)};
                %end
                %temp_grad = lambda{hi,sub}-lambda{hi,kk(sub)};
                %if norm(lambda{hi,sub}-cluster(1,:),2)<norm(lambda{hi,sub}-cluster(2,:),2)
                %if temp_clust1< temp_clust2
                %    temp_grad = lambda{hi,sub}-diag(cluster(1,:));
                %else
                %    temp_grad = lambda{hi,sub}-diag(cluster(2,:));
                %end
                %((temp_clust1<temp_clust2) + features(sub))
                grad_lambda{hi,sub} = ld{index(sub),hi}*diag(diag(-2*(Y{hi}'*(new_A{sub,hi})*Y{hi}) + 2*(Y{hi}')*Y{hi}*(ld{index(sub),hi}*lambda{hi,sub}+C{index(sub),hi})*(Y{hi}')*Y{hi}));
                    %+gamma_cluster*(train_index(sub)==0)*(l>150)*(temp_grad);
                
                
                % update lambda1
                m_lambda{hi,sub} = beta1*past_m_lambda{hi,sub} + (1-beta1)*grad_lambda{hi,sub};
                
                v_lambda{hi,sub} = beta2*past_v_lambda{hi,sub} + (1-beta2)*grad_lambda{hi,sub}.^2;
                v_lambda{hi,sub} = max(v_lambda{hi,sub},past_v_lambda{hi,sub});
                
                %lambda{hi,sub} = lambda{hi,sub} - eta*(m_lambda{hi,sub})./(sqrt(v_lambda{hi,sub})+eps);
                
                past_m_lambda{hi,sub} = m_lambda{hi,sub};
                past_v_lambda{hi,sub} = v_lambda{hi,sub};
                
                
                grad_C{index(sub),hi} = grad_C{index(sub),hi} +diag(diag(-2*(Y{hi}'*(new_A{sub,hi})*Y{hi}) + 2*(Y{hi}')*Y{hi}*(ld{index(sub),hi}*lambda{hi,sub}+C{index(sub),hi})*(Y{hi}')*Y{hi}));
                
                grad_ld{index(sub),hi} = grad_ld{index(sub),hi} +lambda{hi,sub}*diag(diag(-2*(Y{hi}'*(new_A{sub,hi})*Y{hi}) + 2*(Y{hi}')*Y{hi}*(ld{index(sub),hi}*lambda{hi,sub}+C{index(sub),hi})*(Y{hi}')*Y{hi}));
            end
            grad_lambda_avg{hi} = grad_lambda_avg{hi}+invert_r(index(sub))*( diag(diag(-2*(Y{hi}'*(A{sub})*Y{hi}) + 2*(Y{hi}')*Y{hi}*lambda_avg{hi}*(Y{hi}')*Y{hi}))...
                + invert_r(index(sub))*(l>start_hetero)* diag(diag(-2*(Y{hi}'*(new_A_avg{sub,hi})*Y{hi}) + 2*(Y{hi}')*Y{hi}*lambda_avg{hi}*(Y{hi}')*Y{hi})));
        end
        
        m_lambda_avg{hi} = beta1*past_m_lambda_avg{hi} + (1-beta1)*grad_lambda_avg{hi};
        
        v_lambda_avg{hi} = beta2*past_v_lambda_avg{hi} + (1-beta2)*grad_lambda_avg{hi}.^2;
        v_lambda_avg{hi} = max(v_lambda_avg{hi},past_v_lambda_avg{hi});
        past_m_lambda_avg{hi} = m_lambda_avg{hi};
        past_v_lambda_avg{hi} = v_lambda_avg{hi};
        
        
        %for hi=1:hierarchy
  
            if l > start_hetero
                for sub = 1:subjects
                    lambda{hi,sub} = lambda{hi,sub} - 2*eta*(m_lambda{hi,sub})./(sqrt(v_lambda{hi,sub})+eps);
                    temp_lamb = lambda{hi,sub};
                    for lambe=1:k(hi+1)
                        if ((temp_lamb(lambe,lambe))<0)
                            temp_lamb(lambe,lambe) = 0;
                        end
                    end
                    %temp_lamb = temp_lamb*k(hi+1)/sum(sum(temp_lamb));
                    lambda{hi,sub} = temp_lamb;
                    
                end
            end
            
            
        %end
        if l > start_correction
            
            
            for i = 1:scanners
                %grad_C{i,hi} = grad_C{i,hi}/(length(index(index==i)));
                m_C{i,hi} = beta1*past_m_C{i,hi} + (1-beta1)*grad_C{i,hi};
                
                v_C{i,hi} = beta2*past_v_C{i,hi} + (1-beta2)*grad_C{i,hi}.^2;
                v_C{i,hi} = max(v_C{i,hi},past_v_C{i,hi});
                past_m_C{i,hi} = m_C{i,hi};
                past_v_C{i,hi} = v_C{i,hi};
                
                %grad_ld{i,hi} = grad_ld{i,hi}/(length(index(index==i)));
                m_ld{i,hi} = beta1*past_m_ld{i,hi} + (1-beta1)*grad_ld{i,hi};
                
                v_ld{i,hi} = beta2*past_v_ld{i,hi} + (1-beta2)*grad_ld{i,hi}.^2;
                v_ld{i,hi} = max(v_ld{i,hi},past_v_ld{i,hi});
                past_m_ld{i,hi} = m_ld{i,hi};
                past_v_ld{i,hi} = v_ld{i,hi};
            end
        end
        %for hi=1:hierarchy
            
            if l > start_correction
                
                for i=1:scanners
                    %s_C = eigs(step_C{i,hi},1);
                    %C{i,hi} = C{i,hi} - grad_C{i,hi}/(s_C);
                    %C{i,hi} = C{i,hi} - correction*(eta/10)*(m_C{i,hi})./(sqrt(v_C{i,hi})+eps);
                    ld{i,hi} = ld{i,hi} - correction*(eta/10)*(m_ld{i,hi})./(sqrt(v_ld{i,hi})+eps);
                    
                end
            end
        %end
        grad_W{hi} = zeros(k(hi),k(hi+1));
        nnz_val = find(train_index);
        for trav = 1:sum(train_index)
            sub = nnz_val(trav);
            for i=hi:hierarchy
                if (hi==1)
                    grad_W{hi} = grad_W{hi} + invert_r(index(sub))*(l>start_hetero)*(-4*new_A{sub,hi}*W{hi}*(X{i,hi,sub}+X_avg{i,hi}) ...
                        + 4*W{hi}*(X{i,hi,sub}+X_avg{i,hi})*W{hi}'*W{hi}*(X{i,hi,sub}+X_avg{i,hi}))...
                        + invert_r(index(sub))*influe*(-4*new_A{sub}*W{hi}*X_avg{i,hi} ...
                        + 4*W{hi}*X_avg{i,hi}*W{hi}'*W{hi}*X_avg{i,hi});
                else
                    grad_W{hi} = grad_W{hi} +  invert_r(index(sub))*(l>start_hetero)*(-4*Y{hi-1}'*new_A{sub,hi}*Y{hi-1}*W{hi}*(X{i,hi,sub}+X_avg{i,hi}) ...
                        + 4*Y{hi-1}'*Y{hi-1}*W{hi}*(X{i,hi,sub}+X_avg{i,hi})*W{hi}'*Y{hi-1}'*Y{hi-1}*W{hi}*(X{i,hi,sub}+X_avg{i,hi}))...
                        + invert_r(index(sub))*influe*(-4*Y{hi-1}'*new_A{sub}*Y{hi-1}*W{hi}*X_avg{i,hi} ...
                        + 4*Y{hi-1}'*Y{hi-1}*W{hi}*X_avg{i,hi}*W{hi}'*Y{hi-1}'*Y{hi-1}*W{hi}*X_avg{i,hi});
                end
            end
        end
        
        
        m_W{hi} = beta1*past_m_W{hi} + (1-beta1)*grad_W{hi};
        v_W{hi} = beta2*past_v_W{hi} + (1-beta2)*grad_W{hi}.^2;
        v_W{hi} = max(v_W{hi},past_v_W{hi});
        
        %W{hi} = W{hi} - (eta)*(m_W{hi})./(sqrt(v_W{hi})+eps);
        past_m_W{hi} = m_W{hi};
        past_v_W{hi} = v_W{hi};
        
    end
    
    % update gradients
    
    for hi=1:hierarchy
        
        W{hi} = W{hi} - (eta)*(m_W{hi})./(sqrt(v_W{hi})+eps);
        
        temp = W{hi};
        for n_k1=1:k(hi+1)
            temp(:,n_k1) = proj_L1_Linf(squeeze(temp(:,n_k1)),alpha(hi));
            if hi>1
                temp(temp<0) = 0;
            end
        end
        W{hi} = temp;
        if  sum(sum(isnan(W{hi}))) > 0
            return
        end
        
    end
    
end


