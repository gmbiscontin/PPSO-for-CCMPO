clc
clear all
close all

format long

load EUstock40.csv; % import prices
[n,numvar] = size(EUstock40);
rend = (EUstock40(2:end,:) - EUstock40(1:end-1,:))./EUstock40(1:end-1,:);



%inertia_weight_choice = 0; %0->w constant; 1->decreasing
lambda_num = 15;
lambda_val = linspace(0, 1, lambda_num);
runs = 5;
results = zeros(lambda_num,numvar+5);



%% data

media = mean(rend);
variance = cov(rend);
%pigre = 0.049149;



%% PSO setting 

P = 200;
niter = 5000; % numero iterazioni
c1_max = 2.5;
c1_min = 0.5;
c2_max = 2.5;
c2_min = 0.5;
vmaxx = zeros(1,numvar); % vettore di servizio
epsilon1 = 1.0e-003; % parametro di penalizzazione
%w = 0.7298;
w0 = 0.95;
wT = 0.2;
cardinality_constraint = 10;
b = 0.5;
upper_limit = 30;
UB = ones(1, numvar) * upper_limit; %  upper_limit
lower_limit = -5;
LB = zeros(1, numvar)* lower_limit; % lower_limit
max_iter_without_improvement = 20; % Numero massimo di iterazioni senza miglioramenti
iter_without_improvement = 0;
max_weight_value = 0.4; % max-buy contraint

% Inizializzazione della variabile per mantenere traccia della migliore soluzione per ogni lambda
best_fitness_per_lambda = Inf(lambda_num, 1);
best_portfolio_per_lambda = zeros(lambda_num, numvar);


for idx = 1:numel(lambda_val)
    
    lambda = lambda_val(idx);
    best_fitness = Inf;
    best_portfolio = zeros(1, numvar);
    best_risk = Inf;
    best_return = -Inf;

    for iii = 1:runs
        iter_without_improvement = 0;
    
        tic
        %% service vectors and matrices
    
        vmaxx = zeros(1,numvar);
        var_port = zeros(P,1);
        med_port = zeros(P,1);
        fitness_values = zeros(10, 1);
    
        vinc_1 = zeros(P,1); % vincolo di bilancio
        vinc_2 = zeros(P,1); % vincolo di redditività
        app_1 = zeros(P,numvar);
        vinc_3 = zeros(P,1); % x >= 0
        vinc_4 = zeros(P,1);
        vinc_5 = zeros(P,1);
    
    
        %% random initialization
    
        x = rand(P,numvar);
        vx = rand(P,numvar);
        f = ones(P,1)*1.0e+015;
    
    
        %% pbest, gbest and related values of the function;
        pbx = [x f];
        gx = zeros(1,numvar+1);
    
        
        %%
        for k = 1:niter
    
            w_t = (w0 - wT) * ((niter - k) / niter) + wT;
            c1_t = ((c1_min - c1_max) * (k / niter)) + c1_max;
            c2_t = ((c2_max - c2_min) * (k / niter)) + c2_min;
    
            % 1) range for max speed
            for i=1:numvar
                vmaxx(i)=abs(max(x(:,i))-min(x(:,i)));
            end
    
            % 2) calcolo della funzione obiettivo
            for p = 1:P
                
                var_port(p) = x(p,:)*variance*x(p,:)'; % varianza
                med_port(p) = x(p,:)*media'; % media
                vinc_1(p) = abs(sum(x(p,:))-1); % violazione del vincolo di bilancio
                %vinc_2(p) = abs(x(p,:)*media' - pigre); %  violazione del vincolo di redditività
                %for i = 1:numvar
                %    app_1(p,i) = max(0,-x(p,i));
                %end
                vinc_3(p) = sum(app_1(p,:)); % violazione dei vincoli x_i >= 0
                vinc_4(p) = sum(max(0, abs(x(p,:)) - max_weight_value));  % l'abs solo perche ci possono anche essere valori negativi
                vinc_5(p) = -min(sum(x(p,:) > 0)-cardinality_constraint, 0);
                % Enforce cardinality constraint
                if vinc_5(p) < 0        
                    excess_indices = find(x(p,:) > 0, sum(x(p,:) > 0) - cardinality_constraint);        
                % Set the excess assets to zero
                    x(p, excess_indices) = 0;
                end


            end
            % calcolo funzione di fitness
            f_weighted = (lambda*var_port - (1-lambda)*med_port)+(1/epsilon1)*(vinc_1 + vinc_3+ vinc_4+ vinc_5); 
            %f_weighted = lambda*var_port + (1/epsilon1)*(vinc_1 + vinc_2) - (1-lambda)*med_port;    
            %f_weighted = lambda*var_port + (1/epsilon1)*(vinc_1 + vinc_2+vinc_4) - (1-lambda)*med_port;  
            
            % 3) confronto valore della funzione obiettivo con il pbest
            for p = 1:P
                if f_weighted(p) < pbx(p,numvar+1)
                    pbx(p,numvar+1)=f_weighted(p);
                    for i = 1:numvar
                        pbx(p,i) = x(p,i);
                    end
                end
            end
    
            % 4) identificazione particella con migliore posizione
            [minimo,posizione] = min(pbx(:,numvar+1));
            gx(numvar+1) = minimo;
            for i = 1:numvar
                gx(i) = pbx(posizione,i);
            end
    
            % 5) aggiornamento velocità e posizioni
            %if inertia_weight_choice == 1
            %    w = 0.9 - k*0.5/niter;
            %end
            for p = 1:P
                for i = 1:numvar
                    vx(p,i) = w_t*vx(p,i)+c1_t*rand*(pbx(p,i)- x(p,i))+c2_t*rand*(gx(i)- x(p,i));
                    if vx(p,i) > vmaxx(i)
                       vx(p,i) = vmaxx(i);
                    end
                    x(p,i) = x(p,i)+vx(p,i);

                    % Mutation
                    % Calcola il cambiamento della posizione della particella basato sull'iterazione corrente
                    %D = (x(p,i)-0.01*x(p,i)) * (1 - randi([0, 1]))^(1 - k / niter)^b;
                    %flip = randi([0, 1]);
                    %if flip == 0
                    %    flip2 = randi([0, 1]);
                    %    if flip2 == 0 
                    %        x(p,i) = x(p,i)+vx(p,i) + D;
                    %    else
                    %        x(p,i) = x(p,i)+vx(p,i) - D;
                    %    end
                    %else
                    %    x(p,i) = x(p,i)+vx(p,i);
                    %end
                    
                    %x(p,i) = max(LB(i), min(UB(i), x(p,i)));
                    
                end
                
           
                
            end
            if sum(x(p,:) > 0) > cardinality_constraint
                [~, sorted_indices] = sort(abs(x(p,:)), 'descend'); % Sort the elements in descending order
                remove_indices = sorted_indices(cardinality_constraint+1:end); % Select excess elements
                x(p, remove_indices) = 0; % Remove excess elements
            end

            end
            
            % stop criteria after n no improvement iteration
            portfolio = gx(1:end-1)';
            fitness = gx(end);
            if fitness < best_fitness
                best_fitness = fitness;
                best_portfolio = portfolio;
                iter_without_improvement = 0; % Resetta il contatore se c'è un miglioramento
            else
                iter_without_improvement = iter_without_improvement + 1; % Incrementa il contatore se non c'è miglioramento
            end
            % Verifica il criterio di stop
            if iter_without_improvement >= max_iter_without_improvement
                break; % Interrompi l'algoritmo se il criterio di stop è soddisfatto
            end
            
            
        end

    
    
    
    
    tc0 = toc;
    



    portfolio = gx(1:end-1)';
    fitness = gx(end);
    vinc_somma_percentuali = sum(gx(1:end-1)) - 1;
    vinc_reddittivita = gx(1:end-1)*media' - pi;
    vinc_non_negativita = sum(max(0,-gx(1:end-1)'));
  

    %[min_fitness, min_idx] = min(fitness_values); non credo serva perchè
    %voglio salvarle tutte le lambda
    

    % Se la fitness di questa esecuzione è migliore della migliore trovata finora per questo valore di lambda
    if fitness < best_fitness
        % Aggiorna la migliore fitness e il relativo portafoglio per questo valore di lambda
        best_fitness = fitness;
        best_portfolio = portfolio;
    end




    results(idx, 1:numvar) = best_portfolio';
    results(idx, numvar+1) = best_fitness;
    results(idx, numvar+2) = best_portfolio' * variance * best_portfolio;
    results(idx, numvar+3) = media * best_portfolio;
    results(idx, numvar+4) = sum(best_portfolio) - 1;
    results(idx, numvar+5) = tc0;
    results(idx, numvar+6) = lambda;

    %if iii <= 10
    %    figure
    %    plot(converg)
    %    title(['Fitness function - Iteration: ' num2str(iii)])
    %    grid on
    %    axis([0 niter 0 250])
    %    xlabel('Iterations')
    %    ylabel('Fitness')
    %end
end
    





%% true solution

tic

V1 = inv(variance);
e = ones(1,numvar);
alfa = media*V1*media';
beta = media*V1*e';
gamma = e*V1*e';
NUME = (gamma*V1*media' - beta*V1*e') + (alfa*V1*e' - beta*V1*media');
DENO = alfa*gamma - beta^2;

tru_sol = NUME/DENO;

tc1 = toc;



%% results

results = sortrows(results,numvar+1);
dd = dist([results(1,1:numvar);tru_sol']');
dd(2,1); % Euclidean distance
[results(1,1:numvar);tru_sol']'; % pso solution; true solution
%[results(1,numvar+2:end); tru_sol'*variance*tru_sol 0 0]


combined_results = [results; [tru_sol' 0 tru_sol'*variance*tru_sol 0 0 tc1 0]];

% Define number of assets
num_assets = size(results, 2) - 6;

% Create cell array for column names
column_names_results = cell(1, num_assets + 6);

% Generate column names for assets
for i = 1:num_assets
    column_names_results{i} = ['Asset_', num2str(i)];
end

column_names_results{num_assets + 1} = 'Fitness';
column_names_results{num_assets + 2} = 'Portfolio_Variance';
column_names_results{num_assets + 3} = 'Portfolio_Return';
column_names_results{num_assets + 4} = 'Portfolio_Weight_Sum';
column_names_results{num_assets + 5} = 'Execution_Time';
column_names_results{num_assets + 6} = 'Lambda';

%column_names_true_solution = {'True_Solution'};

% Salvataggio dei risultati dell'algoritmo PSO in un file Excel
results_table = array2table(combined_results, 'VariableNames', column_names_results);
results_filename = 'results_names2_lambda.xlsx';
writetable(results_table, results_filename, 'Sheet', 'PSO_Results');



%% plotting

Assets = column_names_results(1,1:num_assets);
port = Portfolio('AssetList', Assets(1:num_assets));

DataSubset = EUstock40(:, 1:num_assets); % Utilizza solo i primi 12 asset, se necessario
port = estimateAssetMoments(port, DataSubset, 'missingdata', true);
port = setDefaultConstraints(port);

figure;
plotFrontier(port);
% Tracciare la frontiera efficiente

hold on;
rendimenti = results(:, numvar+3);
varianze = results(:, numvar+2);
scatter(varianze, rendimenti, 'filled');
title('portafogli ottimi');
xlabel('Varianza');
ylabel('Rendimento atteso');
legend('Frontiera efficiente standard','Portafogli ottimi');
best_portfolio_index = find(rendimenti == min(rendimenti));
best_portfolio_variance = varianze(best_portfolio_index);
best_portfolio_return = rendimenti(best_portfolio_index);
% Imposta i limiti y leggermente più stretti intorno al miglior portafoglio
hold off;


figure;
scatter(varianze, rendimenti, 'filled');
title('portafogli ottimi');
xlabel('Varianza');
ylabel('Rendimento atteso');
legend('Portafogli ottimi');






