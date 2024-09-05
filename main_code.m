clc
clear all
close all
load data_benchmark.mat
fs=250;

%% source_subject 7
num_source_sub = 7;       % zw: choose 7 subjects as the source subject 
all_source_sub = 1:1:35;  % total subjects
source_index = all_source_sub(randperm(numel(all_source_sub),num_source_sub));  

for source_sub_i = 1:1:num_source_sub
    
    source_eeg = squeeze(benchmark(source_index(source_sub_i),:,:,:,:));  %channel sample target trial
    [num_chans, num_samps, num_targs, num_trial_s] = size(source_eeg);
    

    for targ_i = 1:num_targs
        for chan_i = 1:num_chans
            for tri_i = 1:num_trial_s
                x_s(:) = source_eeg(chan_i, :, targ_i, tri_i);
                Wp = [7 90]/(fs/2);   
                Ws = [4 100]/(fs/2);   
                [n, Wp] = cheb1ord(Wp, Ws, 3, 20);  % Chebyshev Type Ⅰ Infinite Impulse Response filter (IIR)
                [B, A] = cheby1(n, 3, Wp);
                sub_datas(chan_i, :, targ_i, tri_i) = filtfilt(B, A, x_s);
            end
        end
    end

    % cue 0.5s + latency 0.14s = 0.64*250 = 160
    filtered_eegs = sub_datas(:, 160:end, :, :); %([48 54 55 56 57 58 61 62 63],:,:,:); %filtered_eeg=sub_data; %channel sample target trial
    source_SSVEPdata(source_sub_i,:,:,:,:) = permute(filtered_eegs,[1 2 4 3]); 
    % source_SSVEPdata 5D 
    %  scr x chan x sam x trial x sti  7x9x1340x6x40
end


%% target_subject 28：select one subject as target (5 block training data + 1 block as test)
tsub = 1:35;
tsub(source_index) = [];  

% pick one target subject from the remaining subjects  
for target_i = 1:numel(tsub) 

    target_eeg = squeeze(benchmark(tsub(target_i),:,:,:,:)); % channel sample target trial
    [num_chans, num_samps, num_targs, num_trial_t] = size(target_eeg);


    for targ_i = 1:num_targs
        for chan_i = 1:num_chans
            for tri_i = 1:num_trial_t
                x_t(:) = target_eeg(chan_i, :, targ_i, tri_i);
                Wp = [7 90]/(fs/2);
                Ws = [4 100]/(fs/2);
                [n,Wp] = cheb1ord(Wp,Ws,3,20);
                [B,A] = cheby1(n,3,Wp);
                sub_datat(chan_i, :, targ_i, tri_i) = filtfilt(B, A, x_t);
            end % trials
        end % channels
    end % targets 

    filtered_eegt = sub_datat(:, 160:end, :, :);%([48 54 55 56 57 58 61 62 63],:,:,:);%filtered_eeg=sub_data;%channel sample target trial
    target_SSVEPdata = permute(filtered_eegt,[1 2 4 3]);
    % target_SSVEPdata 4D 
    % chan x sam x trail x sti 9x1340x6x40

    TW = 0.2: 0.2: 2;   
    TW_p = [50 100 150 200 250 300 350 400 450 500];   
    t_length = 2;

    %% Construct reference signals of sine-cosine waves
    sti_f=[8 9 10 11 12 13 14 15 8.2 9.2 10.2 11.2 12.2 13.2 14.2 15.2 8.4 9.4 10.4 11.4 12.4 13.4 14.4 15.4 8.6 9.6 10.6 11.6 12.6 13.6 14.6 15.6 8.8 9.8 10.8 11.8 12.8 13.8 14.8 15.8];
    Nh=5;  
    for i = 1:1:num_targs
        reference(i,:,:)=refsig(sti_f(i),fs,t_length*fs,Nh);
    end
    
    %% Recognition
    % Estimate classification performance
    is_ensemble = 0; 
    labels = 1:1:num_targs;
    multicov_targ_eeg = permute(target_SSVEPdata, [4,1,2,3]);   %   40x9x1340x6  
    multicov_sour_eeg = permute(source_SSVEPdata, [1,5,2,3,4]); % 7x40x9x1340x6 src x sti x chan x sam x trial

    for tw_length = 1:5        
        
        tic
        accu_all=[];
        itrs_all=[];

        fprintf('Source subject %d %d %d %d %d %d %d, Target subject %d\n', [source_index], tsub(target_i));
        fprintf('tw = %3.2fs\n', TW(tw_length));

        for loocv_i = 1:1:num_trial_t    

            target_traindata = multicov_targ_eeg(:,:,1:TW_p(tw_length),:);
            target_traindata(:,:,:,loocv_i) = [];  
            %%%%%%%%%%%%%%%%%%%%   Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for source_sub_i = 1:1:num_source_sub

                source_traindata = squeeze(multicov_sour_eeg(source_sub_i,:,:,1:TW_p(tw_length),:));  % Nf x Nc x Ns x Nt
                ref = reference(:,:,1:TW_p(tw_length));    % Nf x 2Nh x Ns                
                model_self_t = train_multicov(target_traindata, ref, fs);  % target_traindata  Nf x Nc x Ns x (Nt-1)
                model_self_s{source_sub_i} = train_multicov(source_traindata, ref, fs);
       
                % target template
                % targ_template = squeeze(mean(target_traindata,4));
                for class_i = 1:num_targs
                    for tri_i = 1: size(target_traindata,4)
                        targ_traindata = squeeze(target_traindata(class_i,:,:,tri_i));
                        
                        % transfered template Nf x Ns
                        transfered_template1 = model_self_s{source_sub_i}.W(:,(num_chans+1):2*num_chans)*squeeze(model_self_s{source_sub_i}.trains(class_i,:,:));  % Nf x Ns
                        transfered_template2 = model_self_s{source_sub_i}.W(:,(2*num_chans+1):end)*squeeze(ref(class_i,:,:)); 
                        

                        % transfered filter Nc x Nf
                        t1(tri_i,:,:) = inv(targ_traindata*targ_traindata')*targ_traindata*transfered_template1'; % transfered spatial filter S
                        t2(tri_i,:,:) = inv(targ_traindata*targ_traindata')*targ_traindata*transfered_template2'; % transfered spatial filter T
                   
                                     
                    end
                    T{source_sub_i,class_i,1} = squeeze(mean(t1));  
                    T{source_sub_i,class_i,2} = squeeze(mean(t2));  
                end
            end
            
            % find source weight
            [source_score_normed1,source_score_normed2] = find_source_weight(target_traindata, T, model_self_s,ref);       

            %%%%%%%%%%%%%%%%%%%%%%% Test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            testdata = squeeze(multicov_targ_eeg(:, :,1:TW_p(tw_length),loocv_i));

            estimated = test_lst_targ_multis_score(testdata, model_self_s, model_self_t, T, ref, source_score_normed1,source_score_normed2, is_ensemble);

            % Evaluation
            is_correct = (estimated==labels);
            accs(loocv_i) = mean(is_correct)*100;
            itrs(loocv_i) = itr(num_targs, mean(is_correct), TW(tw_length)+0.5);
            fprintf('\tBlock %d: Accuracy = %5.2f%%, ITR = %2.f bpm\n', loocv_i, accs(loocv_i), itrs(loocv_i));

        end %loocv_i

        accuracy_multicov(target_i, tw_length)= mean(accs);
        itr_multicov(target_i, tw_length)=mean(itrs);

        fprintf('\tMean accuracy = %5.2f%%\n\t', mean(accs));
        toc
    end %TW
end %target sub_i

accuracy_multicov
itr_multicov
mean(accuracy_multicov)
mean(itr_multicov)