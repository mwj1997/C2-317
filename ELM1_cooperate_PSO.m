% initialize the particle swarm
M=20; % the number of individuals
N=3; % the number of parameters
iteration=3;

sigma=zeros(M,N);
a=zeros(M,N);
numda=abs(rand(1,M));

sigma_gbest=zeros(N,N);
a_gbest=zeros(N,N);
numda_gbest=numda(1:3);

sigma_pbest=zeros(M,N);
a_pbest=zeros(M,N);
numda_pbest=rand(1,M);

F_gbest=ones(1,N)*1000;

LS2=(w*X_test)'+b(ones(1,NumberofTestingData),:);
H2=1./(1+exp(-1.*LS2));

for d=1:N
    
     sigma(:,1:d)=rand(M,d); % initialization of d dimension
     a(:,1:d)=rand(M,d); % initialization of d dimension
     numda=rand(1,M);   % initialization of numda
     
     sigma_pbest(:,1:d)=sigma(:,1:d);
     a_pbest(:,1:d)=a(:,1:d);
     numda_pbest=numda;
     F_pbest=ones(1,M)*1000;
     beta=zeros(M,40);
     
    for t=1:iteration
        %% population 1
        for i=1:M       %calculation of the cost function
            [beta2]=ELM1_cost(H,beta(i,:),T_train,sigma(i,1:d),a_pbest(i,1:d),d);
            y2=H2*beta2;
            F(i)=mse(y2'-T_test)+numda_pbest(i)*abs(sum(a_pbest(i,1:d).*a_pbest(i,1:d))-1);
        end
        for i=1:M
            if F_pbest(i)>F(i)
                F_pbest(i)=F(i);
                sigma_pbest(i,1:d)=sigma(i,1:d);
            end
             if F_gbest(d)>F(i)
                F_gbest(d)=F(i);
                sigma_gbest(d,1:d)=sigma(i,1:d);
                a_gbest(d,1:d)=a(i,1:d);
            end
        end
     clear y2;      
    thres1=sigma(:,1:d);
    thres2=mean(thres1);
    thres3=(1/(M-1)).*sum((thres1-thres2(ones(1,M),:)).*(thres1-thres2(ones(1,M),:)));
    
    for i=1:M
        r1=rand(1,d).*thres3;
        r2=rand(1,d).*thres3;
        sigma(i,1:d)=sigma(i,1:d)+r1.*(sigma(i,1:d)-sigma_pbest(i,1:d))+r2.*(sigma(i,1:d)-sigma_gbest(d,1:d));
    end
    
    %% population 2
    for i=1:M       %calculation of the cost function
            [beta2]=ELM1_cost(H,beta(i,:),T_train,sigma_pbest(i,1:d),a(i,1:d),d);
            y2=H2*beta2;
            F(i)=mse(y2'-T_test)+numda_pbest(i)*abs(sum(a(i,1:d).*a(i,1:d))-1);
    end
        
    for i=1:M
            if F_pbest(i)>F(i)
                F_pbest(i)=F(i);
                a_pbest(i,1:d)=a(i,1:d);
            end
             if F_gbest(d)>F(i)
                F_gbest(d)=F(i);
                sigma_gbest(d,1:d)=sigma(i,1:d);
                a_gbest(d,1:d)=a(i,1:d);
            end
    end
      
    clear y2;      
    thres1=a(:,1:d);
    thres2=mean(thres1);
    thres3=(1/(M-1)).*sum((thres1-thres2(ones(1,M),:)).*(thres1-thres2(ones(1,M),:)));
    
    for i=1:M
        r1=rand(1,d).*thres3;
        r2=rand(1,d).*thres3;
        a(i,1:d)=a(i,1:d)+r1.*(a(i,1:d)-a_pbest(i,1:d))+r2.*(a(i,1:d)-a_gbest(d,1:d));
    end
    
    %% population 3
     for i=1:M       %calculation of the cost function
            [beta2]=ELM1_cost(H,beta(i,:),T_train,sigma_pbest(i,1:d),a_pbest(i,1:d),d);
            y2=H2*beta2;
            F(i)=mse(y2'-T_test)+numda(i)*abs(sum(a(i,1:d).*a(i,1:d))-1);
     end
    
     for i=1:M
            if F_pbest(i)>F(i)
                F_pbest(i)=F(i);
                numda_pbest(i)=numda(i);
            end
             if F_gbest(d)>F(i)
                F_gbest(d)=F(i);
                numda_gbest(d)=numda(i);
            end
     end
    
    thres1=numda;
    thres2=mean(thres1);
    thres3=(1/(M-1)).*sum((thres1-thres2(1,ones(1,M))).*(thres1-thres2(1,ones(1,M))));
    for i=1:M
        r1=rand(1,1).*thres3;
        r2=rand(1,1).*thres3;
        numda(i)=numda(i)+r1.*(numda(i)-numda_pbest(i))+r2.*(numda(i)-numda_gbest(d));
    end
    numda=abs(numda);
    end
end