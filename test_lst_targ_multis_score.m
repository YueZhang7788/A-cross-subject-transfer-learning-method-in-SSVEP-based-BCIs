function results = test_lst_targ_multis_score(eeg, model_self_s, model_self_t, T, ref, source_score_normed1,source_score_normed2,is_ensemble)

%% 
% Inputs 
%  ** eeg: single-trial test data from the target subject  #stimulus x #channels x #samples 
%  ** model_self_s: model_self_s{source subject}."trains" "W" "fs" "num_targets"
%  ** model_self_t: model_self_s."trains" "W" "fs" "num_targets"
%  ** T: transferd spatial filters  T{source subject, class, 1(S) / 2(T)}
%  ** ref: artificial reference  #stimulus x #2harmonics x #samples 
%  ** source_score_normed1: ?
%  ** source_score_normed2: ?
%  ** is_ensemble: 1/0
%  Outputs
%  ** results: one number from the label vector.

[num_targs, num_chans, num_smpls, ~] = size(eeg);    % ?? eeg should be 3D here 
for targ_i = 1:1:num_targs
    test_tmp = squeeze(eeg(targ_i, :, :));
    for class = 1:1:num_targs
        ref_tmp =  squeeze(ref(class,:, :));
        for source_sub = 1:size(T,1)
            inter_t1 = squeeze(T{source_sub,class,1})';  %S
            inter_t2 = squeeze(T{source_sub,class,2})';  %T

            % from source subject
            transfered_template1 = model_self_s{source_sub}.W(:,(num_chans+1):2*num_chans)*squeeze(model_self_s{source_sub}.trains(class,:,:));
            transfered_template2 = model_self_s{source_sub}.W(:,(2*num_chans+1):end)*ref_tmp;
            r1_tmp = corrcoef(inter_t1*test_tmp, transfered_template1); 
            rr_s(source_sub,class, 1) = r1_tmp(1,2);
            r2_tmp = corrcoef(inter_t2*test_tmp, transfered_template2); 
            rr_s(source_sub,class, 2) = r2_tmp(1,2);
            
            % for i = 1:2
            %   r_s(source_sub,class) = r_s(source_sub,class) + sign(rr_s(source_sub,class,i))*rr_s(source_sub,class,i)^2;
            % end
        end

        % r_s1 = source_score_normed1*squeeze(rr_s(:,1));
        % r_s2 = source_score_normed2*squeeze(rr_s(:,2));
        r_s1 = source_score_normed1(:,class)'*squeeze(rr_s(:,class,1));
        r_s2 = source_score_normed2(:,class)'*squeeze(rr_s(:,class,2));
        % r_s1 = mean(squeeze(rr_s(:,1)),1);
        % r_s2 = mean(squeeze(rr_s(:,2)),1);

        rho_s(class) = sign(r_s1)*(r_s1)^2 + sign(r_s2)*(r_s2)^2;
        % rho_s = sum(r_s,1)/size(T,1);

        % from target subject
        if ~is_ensemble
            targ_template = squeeze(model_self_t.trains(class,:,:));
            intra_w1 = squeeze(model_self_t.W(class, 1:num_chans))';
            intra_w2 = squeeze(model_self_t.W(class, num_chans+1:2*num_chans))';
            intra_w3 = squeeze(model_self_t.W(class, 2*num_chans+1:end))';
        else %ensemble
            targ_template = squeeze(model_self_t.trains(class,:,:));
            intra_w1 = squeeze(model_self_t.W(:, 1:num_chans))';
            intra_w2 = squeeze(model_self_t.W(:, num_chans+1:2*num_chans))';
            intra_w3 = squeeze(model_self_t.W(:, 2*num_chans+1:end))';
        end

        r3_tmp = corrcoef(test_tmp'*intra_w1, targ_template'*intra_w2); 
        rr_t(class,1) = r3_tmp(1,2);
        r4_tmp = corrcoef(test_tmp'*intra_w1, ref_tmp'*intra_w3); 
        rr_t(class,2) = r4_tmp(1,2);
        %[w,y,r3] = cca(test_tmp,ref_tmp);  rr(class_i,3) = max(r3);
        r_t(class) = 0;
        for i = 1:2
            r_t(class) = r_t(class) + sign(rr_t(class,i))*rr_t(class,i)^2;
        end
        r(class) = rho_s(class) + r_t(class);
    end % class_i

    [~, tau] = max(r);
    results(targ_i) = tau;

end % targ_i