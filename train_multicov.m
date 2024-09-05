function model = train_multicov(eeg, ref, fs)
%%
% Inputs:
%   eeg   - training data           4D sti x chan x samples x trials             
%   ref   - the artificial template 3D sti x 2*harmonics x samples 
%   fs    - sampling rate 
% Outputs:
%   model - structe data
%            'trains'：individual template  3D  sti x chan x samples
%            'W'     : the eigenvector matrix of inv(Q)*S
%            'fs'    : sampling rate     
%            'num_targs' 


[num_targs, num_chans, ~, ~] = size(eeg);
 num_har = size(ref,2);

trains = squeeze(mean(eeg,4));  
W = zeros(num_targs, num_chans+num_chans+num_har);   

% calculate spatial filters for different targets 
for targ_i = 1:1:num_targs
    eeg_tmp = squeeze(eeg(targ_i, :, :, :));     % 3D Nc x Ns x Nt
    trains_tmp = squeeze(trains(targ_i, :, :));  % 2D Nc x Ns
    ref_tmp = squeeze(ref(targ_i, :, :));        % 2D Nh x Ns
    w_tmp = multicov(eeg_tmp, trains_tmp, ref_tmp);
    W(targ_i, :) = w_tmp(:,1);
end % targ_i

U = W(:,1:num_chans);
V = W(:,num_chans+1:2*num_chans);
Z = W(:,2*num_chans+1:end);

model = struct('trains', trains, ...
               'W', W,...
               'U', U,...
               'V', V,...
               'Z', Z,...
               'fs', fs, ...
               'num_targs', num_targs);

%% %%%%%%%%%%%%%%%%%%%%%% multicov() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function W = multicov(eeg, trains, ref)

[num_chans, num_smpls, num_trials]  = size(eeg);
[num_har, ~]  = size(ref);

%% calculate S
% calculate S11
S = cell(3);
S{1,1} = zeros(num_chans);
for trial_i = 1:num_trials-1
    x1 = squeeze(eeg(:,:,trial_i));
    x1 = bsxfun(@minus, x1, mean(x1,2));
    for trial_j = trial_i+1:num_trials
        x2 = squeeze(eeg(:,:,trial_j));
        x2 = bsxfun(@minus, x2, mean(x2,2));
        S{1,1} = S{1,1} + x1*x2' + x2*x1';    % zw: divide (Nt-1）？？？？
    end % trial_j
end % trial_i

% calculate S12 and S21
S{1,2} = zeros(num_chans);
S{2,1} = zeros(num_chans);
trains = bsxfun(@minus, trains, mean(trains,2));
for trial_i = 1:1:num_trials
    x1 = squeeze(eeg(:,:,trial_i));
    x1 = bsxfun(@minus, x1, mean(x1,2));
    S{1,2} = S{1,2} + x1*trains';
end % trial_i
S{2,1} = S{1,2}';

% calculate S13 and S31
S{1,3} = zeros(num_chans,num_har);
S{3,1} = zeros(num_chans,num_har);
ref = bsxfun(@minus, ref, mean(ref,2));
for trial_i = 1:1:num_trials
    x1 = squeeze(eeg(:,:,trial_i));
    x1 = bsxfun(@minus, x1, mean(x1,2));
    S{1,3} = S{1,3} + x1*ref';
end % trial_i
S{3,1} =  S{1,3}';

% calculate S22
S{2,2} = zeros(num_chans,num_chans);
S{2,2} = trains*trains';
% calculate S23 and S32
S{2,3} = zeros(num_chans,num_har);
S{3,2} = zeros(num_chans,num_har);
S{2,3} = trains*ref';
S{3,2} = S{2,3}';
% calculate S33
S{3,3} = zeros(num_har,num_har);
S{3,3} = ref*ref';
S = cell2mat(S);
%% Calculate Q
Q0 = cell(3,1);
Q0{1} = zeros(num_chans, num_chans);
UX = reshape(eeg, num_chans, num_smpls*num_trials);
UX = bsxfun(@minus, UX, mean(UX,2));
Q0{1} = UX*UX';
Q0{2} = zeros(num_chans, num_chans);
Q0{2} = trains*trains';
Q0{3} = zeros(num_har, num_har);
Q0{3} = ref*ref';
Q = blkdiag(Q0{1}, Q0{2}, Q0{3});

%% Calculate W
[W,~] = eigs(S, Q);




