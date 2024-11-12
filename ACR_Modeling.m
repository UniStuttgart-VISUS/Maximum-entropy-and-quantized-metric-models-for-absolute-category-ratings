%% Maximum entropy and quantized metric models for absolute category ratings
% Code for our paper "Maximum entropy and quantized metric models for absolute 
% category ratings." 
%% 
% * *Authors:* Saupe, D., Rusek, K., Hägele, D., Weiskopf, D., & Janowski, L.
% * *Journal*: <http://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97?source=authoralert 
% IEEE Signal Processing Letters>
% * *Publication Date*: 2024
% * *Volume*: 31 
% * *On Page(s)*: 2970-2974
% * *Print ISSN*: 1070-9908
% * *Online ISSN*: 1558-2361
% * *Digital Object Identifier*: <https://ieeexplore.ieee.org/document/10716765?source=authoralert 
% 10.1109/LSP.2024.3480832>
% * *Communicating author:* Dietmar Saupe, dietmar.saupe@uni-konstanz.de
%% 
% Copyright (C) 2024 by Saupe, D., Rusek, K., Hägele, D., Weiskopf, D., & Janowski, 
% L. <dietmar.saupe@uni-konstanz.de>
% 
% Permission to use, copy, modify, and/or distribute this software for any purpose 
% with or without fee is hereby granted.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
% REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY 
% AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, 
% INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM 
% LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR 
% OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
% OF THIS SOFTWARE.
% 
% The program reads either one of the two datasets KonIQ-10k or VQEG-HDTV and 
% then computes the maximum entropy and quantized metric models for all the 
% ACR distributions in the datasets. This reproduces (among other things) the 
% values in the columns for AIC and G-test in Tables I and II.

clear; close all; clc;
rng (0);

%% Select the model types for ACR probability distributions

% list of available ACR probability distribution types
dlist = ["normal", "logistic", "beta", "kumaraswamy", "logit-logistic", "maxentropy", "GSD"];
setGSD ("gsd_prob_vectors.csv");        % reads table for the GSD distribution
% select any subset of the model types for execution
dlist = ["normal", "logistic", "beta"];
%% Select and import CSV file of an ACR dataset

csvlist = ["KonIQ-10k.csv", "VQEG-HDTV.csv"];  % list of ACR datasets
% column 1 = id of stimulus, cols 2:6 = the counts of the 5-level ACR ratings 1 to 5
csvfile = csvlist(1);        % choose one of the datasets
M = readtable(csvfile);        
head(M)
sample_size = sum(M{:,2:6},2); % total number of ratings for each stimulus
A = M{:,2:6} ./ sample_size;   % create probability vectors by normalizing
n = size(A,1);                 % number of rows (stimuli) in dataset
fprintf ("Imported ACR dataset is %s with %d rows.\n", csvfile, n);
%% Loop over all model types and distributions

nmodels = length(dlist);    % number of models of distributions
rows = n * nmodels;         % Number of rows in output structure
cols = 8;                   % Columns: [image name, ACR ratings, model name, par, pmodel, nll, gtest, pvalue]
C = cell(rows,cols);        % Create a cell array for the output file

nll = zeros(nmodels,n);
gtest = zeros(nmodels,n);
pvalue = zeros(nmodels,n);

for model = 1:nmodels
    % Compute the cumulative distributions of G-test statistic for the data and the data bootstrapped from the models
    distribution = dlist(model);
    fprintf ("Starting with distribution model: %s\n", distribution);
    %
    tic
    for item = 1:n
        if (mod(item,100)==0), fprintf ("%5d items\n", item); end
        p = A(item,1:5) + eps;              % select prob vectors from the dataset one by one
        switch distribution
            case 'maxentropy'
                [par, pmodel, nll(model,item)] = HmaxPV (p); 
            otherwise
                [par, pmodel, nll(model,item)] = solve_opt (p, distribution);
        end
        nll(model,item) = sample_size(item) * nll(model,item);      % multiply by sample size to get the true NLL
        gtest(model,item) = - 2 * sample_size(item) * (sum(p.*log(pmodel)) - sum(p.*log(p)));       % G-test likelihood ratio
        pvalue(model,item) = 1 - chi2cdf(gtest(model,item),2);                                      % p-value from chi2 with df=2

        % output in cell array
        row = (model-1) * n + item;     % row of output cell array
        C(row,:) = {M{item,1}, M{item,2:6}, dlist(model), par, pmodel, nll(model,item), gtest(model,item), pvalue(model,item)};
    end
    T = toc;

    % Collect statistics: Akaike and Bayesian Information Criterion, means of gtest, nll and likelihood
    AIC(model) = (2*n)*2      + 2*sum(nll(model,:));    % AIC = k*2 + 2*NLL, where k = 2*n = number of parameters
    BIC(model) = (2*n)*log(n) + 2*sum(nll(model,:));    % BIC = k*log(n) - 2*log(likelihood)
    mean_gtest(model) = sum(gtest(model,:))/n;
    mean_nll(model) = sum(nll(model,:))/n;
    mean_likelihood(model) = sum( exp(-nll(model,:)./sample_size(:)') )/n;   % mean of the exponential of the expected log-likelihood
    fprintf("AIC %7.1f   BIC %7.1f   Mean G-test %7.5f   Mean NLL %7.5f   Mean likelihood %7.5f\n", ...
        AIC(model), BIC(model), mean_gtest(model), mean_nll(model), mean_likelihood(model));
    fprintf ("Finished in %.2f secs with distribution model <%s> for %d stimuli in %s.\n", T, distribution, n, csvfile);
    fprintf("---------------------------------------------------------------------------------------");
end
%% Generate Excel output file 

filename = sprintf("out %s %s.xlsx", csvfile, datetime);
TC = cell2table(C,...
    "VariableNames",["stimulus" "ratings" "model" "pars" "probs" "nll" "gtest" "pval"]);
writetable(TC,filename);
%% Figure for G-test cdfs

Model_plot = true;
if Model_plot
    fprintf("Critical G-test value at p = 0.05 is %5.2f\n", chi2inv(0.95,2));
    x = zeros(nmodels,n);
    y = zeros(nmodels,n);
    figure
    for model = 1:nmodels
        [x(model,:), y(model,:)] = samplecdf (gtest(model,:));
        p = plot (x(model,:), y(model,:)); hold on
        p.LineWidth = 1.5;
    end
    % also plot the chi2 distribution with 2 degrees of freedom
    p = plot (x(1,:), chi2cdf(x(1,:),2), '--'); hold on
    string = sprintf("%s: G-test CDFs for distribution models", csvfile);
    title(string)
    p.LineWidth = 1.5;
    xlabel('G-Test'); xlim([0 10]); 
    ylabel('CDF'); ylim([0 1.0]); 
    legend([dlist,"chi square"],'Location','southeast');
    hold off
end
%% Solver by optimization

function [par, pvout, nll] = solve_opt (pvin, distribution)
%-----------------------------------------------------------------------
% Interior point solver fmincon from Matlab
% settings for fmincon optimization
DisplayOption = 'none';     % fminunc options: none, iter, notify, final, iter-detailed
nstimulioptions = optimoptions('fmincon');
options = optimoptions(@fmincon, ...
    'MaxIter',1000, ...
    'MaxFunEvals', 10000, ...
    'Algorithm','interior-point',...        %interior-point, sqp
    'SpecifyObjectiveGradient',false, ...
    'CheckGradients', false, ...
    'TolFun',1e-6, ...
    'TolX',1e-6, ...
    'Display', DisplayOption);
% list of constraints
A = [];
b = [];
Aeq = [];
beq = [];
nonlcon = [];
%
switch distribution
    case 'normal'
        delta = 20;
        lb = [1-delta,0];
        ub = [5+delta,100];
    case 'logistic'
        delta = 20;
        lb = [1-delta,0];
        ub = [5+delta,100];
    case 'beta'
        lb = [0,0];
        ub = [20,20];
    case 'kumaraswamy'
        lb = [0,0];
        ub = [20,20];
    case 'logit-logistic'
        lb = [0,0];
        ub = [1,20];
    case 'GSD'
        lb = [1,0];
        ub = [5,1];
end
% function to be minimized
fun = @(x)opt (x, pvin, distribution);
%
% do the optimization
% sometimes the random initial guess leads to convergence to a suboptimal
% local MLE maximum. This is detected by the check if the normalized gtest 
% (=gtest/ratings) is greater than 1. 
% In that condition, a new random initial guess is tried (up to 10 times).
ngtest = inf; iter = 1;
while ngtest > 1.0 & iter <= 10
    x0 = init_solution (distribution, pvin);
    [par,nll,exitflag,output] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
    pvout = ACR(par,distribution) + eps;
    if exitflag <= 0
        fprintf("Exit fmincon in solve_opt (psi, rho). flag = %d\n", exitflag);
        output
        fprintf (" pvin %f %f %f %f %f    entropy %f\n", pvin, - sum(pvin.*log(pvin)));
        fprintf ("pvout %f %f %f %f %f  x-entropy %f\n", pvout, - sum(pvin.*log(pvout)));
    end
    ngtest = - 2 * (sum(pvin.*log(pvout)) - sum(pvin.*log(pvin)));       % normalized G-test
    if (iter==1 & ngtest>1)
        fprintf("Suboptimal local minimum. ngtest %.2f nll %.2f. Next: ", ngtest, nll); 
    end
    if (iter>1 & ngtest>1),
        fprintf("ngtest %.2f  nll %.2f.  Next: ", ngtest,  nll);
    elseif (iter>1 & ngtest<=1)
        fprintf("ngtest %.2f  nll %.2f. OK: Accepted (%d trials).\n", ngtest,  nll, iter);
    end
    iter = iter + 1;
end
pvout = ACR(par,distribution) + eps;
end
%% Setup random initial conditions

function x0 = init_solution (distribution, pvin)
switch distribution
    case 'normal'
        x0(1) = 1 + 4*rand;
        x0(2) = 3*rand;
    case 'logistic'
        x0(1) = 1 + 4*rand;
        x0(2) = 3*rand;
    case 'beta'
        x0(1) = rand;
        x0(2) = rand;
    case 'kumaraswamy'
        x0(1) = rand;
        x0(2) = rand;
    case 'logit-logistic'
        x0(1) = rand;
        x0(2) = 3*rand;
    case 'GSD'
        [x0(1), ~, x0(2)] = SOS(pvin);
end
end
%% Loss function

function nll = opt (x, pv, distribution)
% x = 2d parameter vector
% pv = prob vector
model = ACR(x,distribution)+eps;  % add eps to avoid log(0);
nll = -sum(pv.*log(model));
end
%% ACR probability distribution

function p = ACR(x,distribution)
% x = 2d parameter vector
switch distribution
    case 'normal'
        thresholds = [1.5,2.5,3.5,4.5];
        pcdf = normcdf(thresholds,x(1),x(2));
    case 'logistic'
        thresholds = [1.5,2.5,3.5,4.5];
        pcdf = logisticcdf(thresholds,x(1),x(2));
    case 'beta'
        thresholds = [0.2,0.4,0.6,0.8];
        pcdf = betacdf(thresholds,x(1),x(2));
    case 'kumaraswamy'
        thresholds = [0.2,0.4,0.6,0.8];
        pcdf = kumaraswamycdf(thresholds,x(1),x(2));
    case 'logit-logistic'
        thresholds = [0.2,0.4,0.6,0.8];
        pcdf = logitlogisticcdf(thresholds,x(1),x(2));
    case 'GSD'
        p = getGSD(x(1),x(2));
        return;
    otherwise
        warning('Unexpected distribution type in ACR(x,distributon).')
end
p(1) = pcdf(1) - 0;
p(2) = pcdf(2) - pcdf(1);
p(3) = pcdf(3) - pcdf(2);
p(4) = pcdf(4) - pcdf(3);
p(5) =       1 - pcdf(4);
end
%% $(\psi,var,\rho)$ for a probability vector (from the SOS hypothesis) 

function  [psi, var, rho] = SOS(p)
% Input p = probability row vector of dimension 5
psi = p*[1;2;3;4;5];                            % mean
var = p*(([1;2;3;4;5]-psi).^2);                 % variance
vmin = (ceil(psi)-psi)*(psi-floor(psi));        % minimal variance
vmax = (psi-1)*(5-psi);                         % maximal variance
if (vmax > vmin)
    rho = (vmax - var)/(vmax - vmin);           % rho = 1 -> minimal variance
else
    rho = 0.5;
end
end
%% Compute cumulative distribution function from random samples

function [x, cdf] = samplecdf (data)
% Compute cumulative distribution function for a set of numbers
% Input: data = row vector of real numbers
% Output: cdf as  cdf(x) = proportion of data <= x
x = sort(data);
cdf = (1:length(data))/length(data);
end
%% Kumaraswamy distribution
% Support on [0,1] with cdf $1 -(1-x^a)^b$

function cdf = kumaraswamycdf(x,a,b)
cdf = 1 - (1-x.^a).^b;
end
%% Logistic distribution

function cdf = logisticcdf(x,mu,s)
cdf = 1./(1 + exp(-(x-mu)./s));
end
%% Logit-Logistic distribution

function cdf = logitlogisticcdf(x,mu,s)
cdf = 1./(1 + ((mu*(1-x))./(x*(1-mu))).^(1/s));
end
%% HmaxPV - Maximum entropy probability vector

function [par, pvout, nll] = HmaxPV (pvin)
%-----------------------------------------
% This function outputs psi (mean) and rho (complementary normalized variance) 
% of the maximum entropy probabilty distribution pvout which minimizes
% the negative log-likelihood for the input distribution pvin.
% It also computes the corresponding maximum entropy probability
% distribution pvout.
% By means of a theorem (not contained in the paper), the mean and complementary normalized
% variance of the solution are equal to those of the input distribution (oracle).
% So pvout has the same mean and variance as pvin.
% The main part of this function therefore computes the maximum entropy
% probabilty distribution for the given input mean and complementary normalized
% variance.

[psi, ~, rho] = SOS(pvin);
par = [psi, rho];      % maximum entropy model parameters are psi and rho (Oracle)

% Interior point solver fmincon from Matlab
% settings for fmincon optimization
DisplayOption = 'none';     % fminunc options: none, iter, notify, final, iter-detailed
nstimulioptions = optimoptions('fmincon');
options = optimoptions(@fmincon, ...
    'MaxIter',500, ...
    'MaxFunEvals', 10000, ...
    'Algorithm','interior-point',...        %interior-point, sqp
    'SpecifyObjectiveGradient',true, ...
    'CheckGradients', false, ...
    'TolFun',1e-8, ...
    'TolX',1e-8, ...
    'ConstraintTolerance', 1e-5, ...
    'Display', DisplayOption);
% lower and upper bounds
lb = [0,0,0,0,0];
ub = [1,1,1,1,1];
% list of linear constraints
A = [];
b = [];
Aeq = [1,1,1,1,1];
beq = [1];
% function for nonlinear equality constraints
mycon = @(x)constraints(x, psi, rho);

% do the optimization
fun = @(x)optH(x);                      % negative entropy shall be minimized
p0 = diff([0, sort(rand(1,4)), 1]);     % initialization: random ACR probability vector
% Another good initialization is p0 = [0.2, 0.2, 0.2, 0.2, 0.2];
% Initializations near the boundary of the domain may fail to give convergence.
[pvout,fval,exitflag,output] = fmincon(fun,p0,A,b,Aeq,beq,lb,ub,mycon,options);
if exitflag <= 0
    fprintf("Exit fmincon in HmaxPV (psi, rho). flag = %d\n", exitflag);
    output
    fprintf (" pvin %f %f %f %f %f    entropy %f\n", pvin, - sum(pvin.*log(pvin)));
    fprintf ("pvout %f %f %f %f %f  x-entropy %f\n", pvout, - sum(pvin.*log(pvout)));
end
pvout = pvout + eps;                    % add machine unit to prevent log(0)
nll = - sum(pvin.*log(pvout));
end

function analyse (p, text)              % only for debugging
    [psi, ~, rho] = SOS(p);
    [neg_entropy, grad] = optH (p);
    fprintf ("%s psi/rho %7.5f %7.5f  p %f %f %f %f %f  entropy  %f\n", text, psi, rho, p, -neg_entropy);
end

function [c, ceq] = constraints (p, psi_, rho_)
% c is an array of nonlinear inequality contraints
c = [];
% ceq is an array of two nonlinear equality contraints
% ceq(1) and ceq(2) ensure that p has the required psi and rho
[psi, ~, rho] = SOS(p);
ceq(1) = psi_ - psi;
ceq(2) = rho_ - rho;
end

function [neg_entropy, grad] = optH (p)
% negative entropy to be minimized to obtain maximum entropy
p0 = p(p>0);
neg_entropy = sum(p0.*log(p0));         % negative entropy
if nargout > 1                          % gradient required (column vector!)
    grad = (log(p+eps)+1)';
end
end
%% GSD - Table lookup for the GSD
% The GSD model is comuted from a large table of precomputed models 

function setGSD (csvfile)
%-----------------------------------------------------------------------
global GSDarray;
% Interpretation of GSDarray(i,j,k)
% GSDarray(1,j,k) = psi = 1 + (k-1)*4/1000
% GSDarray(2,j,k) = rho = (j-1)*100
% GSDarray(3:7,j,k) = pv(1:5) for parameters (psi,rho)
T = readtable(csvfile);                     % read csv file with GSD table info
T(1,:) = [];                                % remove table header row
GSDarray = table2array(T)';
GSDarray = reshape(GSDarray,7,101,[]);      % GSDarray has 2+5 columnlength, 101 rowlength, and 1001 pages
end

function pv = getGSD (psi, rho)
%-----------------------------------------------------------------------
global GSDarray;
if ~isreal(psi) || ~isreal(rho)
    psi,rho
end

if (psi<1 || psi>5 || rho<0 || rho>1)
    fprintf ("Error in getGSD %f %f \n", psi, rho);
    psi = min([max([psi,1]),5]);
    rho = min([max([rho,0]),1]);
end
kfloat = 1 + 250*(psi-1);
if (kfloat < 1)
    fprintf("getGSD: kfloat < 1. psi %f\n", psi);
    kfloat = 1;
end
k = (floor(kfloat)); dk = kfloat - k;
if (k==1001)
    k = 1000; dk = 1.0;
end
jfloat = 1 + 100*rho;
if (jfloat < 1)
    fprintf("getGSD: jfloat < 1. rho %f\n", psi);
    jfloat = 1;
end
j = (floor(jfloat)); dj = jfloat - j;
if (j==101)
    j = 100; dj = 1.0;
end
if (j<1)
    fprintf("getGSD: j < 1.  %d\n", j);
    j = 1;
end

%pv = GSDarray(3:7,j,k);             % use this to turn off bilinear interpolation
pv = (1-dj)*(1-dk)*GSDarray(3:7,j,k) + dj*(1-dk)*GSDarray(3:7,j+1,k) + ...
    (1-dj)*dk*GSDarray(3:7,j,k+1) + dj*dk*GSDarray(3:7,j+1,k+1);
pv = pv'+eps;                        % returns a row vector with bilinear interpolation
end
