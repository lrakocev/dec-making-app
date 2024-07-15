function make_dec_timing_map(appr_table, tasktype, maptype)

    figure
    cost_levels = 1:1:4;
    reward_levels = 1:1:4;
    
    observed_p_appr = zeros(4);
    rs = repelem(reward_levels,1,length(reward_levels))';
    cs = repmat(cost_levels,1,length(cost_levels))';
    ps = zeros(length(cost_levels)*length(reward_levels),1);
    
    i = 1;
    for r=1:length(reward_levels)
        for c=1:length(cost_levels)
            if isequal(maptype, "decision_timing")
                ps(i) = mean(appr_table(appr_table.real_c == c & appr_table.real_r == r,:).decision_timing);
            elseif isequal(maptype, "num_inc_in_eng")
                ps(i) = mean(appr_table(appr_table.real_c == c & appr_table.real_r == r,:).big_ups);
            elseif isequal(maptype, "num_local_maxima")
                ps(i) = mean(appr_table(appr_table.real_c == c & appr_table.real_r == r,:).num_maxes);
            end
            observed_p_appr(c,r) = ps(i);
            i = i+1;
        end
    end

    max_val = max(ps);
    min_val = min(ps);
    
    syms R C
    
    % Normal
    imagesc(observed_p_appr);
    %cMap = [interp1(0:1,[0.64 0.08 0.18; 1 1 1],linspace(0,1,100)); ones([100,3]); interp1(0:1,[1 1 1; 0.47 0.67 0.19],linspace(0,1,100))];
    colormap("default");
    cb = colorbar;
    cb.Ticks = [min_val 0.5*(max_val + min_val) max_val];
    caxis([min_val max_val]);
    hold on
    ylabel(cb,'approach rate')
    set(gca,'xtick',[], 'ytick',[], 'FontSize',20, 'YDir','normal');
    xlabel('reward')
    ylabel('cost')
    title("3D Psychometric fun. for task: " + tasktype)
    fighandle = gcf;
    hold off
    saveas(fighandle,strcat(maptype,'_map_', tasktype,'.fig'))

end
