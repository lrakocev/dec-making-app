function bar_plots_by_trial(tasks,tables,bar_type,save_to,is_hr)

bars = [];
errbars = [];
legend_entry = [];
for i = 1:length(tasks)
    tit = tasks{i};
    table = tables{i};
    if is_hr
        curr_table = table(str2double(table.number_of_timesteps(:)) > 9, :);
    else
        curr_table = table;
    end
    
    legend_entry = [legend_entry; tit + ": n = " + string(height(curr_table))];
    
    if isequal(bar_type, "timing")
        curr_table.decision_timing = get_arr(curr_table.decision_timing);
        curr_table = curr_table(curr_table.decision_timing ~= 0, :);
        vals = curr_table.decision_timing;
    elseif isequal(bar_type, "num inc in eng")
       vals = get_arr(curr_table.big_ups);
    elseif isequal(bar_type, "num dec in eng")
        vals = get_arr(curr_table.big_dips); 
    elseif isequal(bar_type, "num changes in eng")
         vals = get_arr(curr_table.big_ups) +  get_arr(curr_table.big_dips);
    elseif isequal(bar_type, "num local maxima")
         vals = get_arr(curr_table.num_maxes);
    elseif isequal(bar_type, "num local minima")
         vals = get_arr(curr_table.num_mins);
    elseif isequal(bar_type, 'avg eng')
         vals = get_arr(curr_table.avg_eng);
    elseif isequal(bar_type, 'eng around dec')
        vals = get_arr(curr_table.eng_around_dec);
    elseif isequal(bar_type, 'direction')
        vals = get_arr(curr_table.direction);
    elseif isequal(bar_type, 'percent max hr')
        vals = get_arr(curr_table.percent_max_hr);
    elseif isequal(bar_type, 'percent min hr')
        vals = get_arr(curr_table.percent_min_hr);
    elseif isequal(bar_type, 'total change')
        max_vals = get_arr(curr_table.percent_max_hr);
        min_vals = get_arr(curr_table.percent_min_hr);
        vals = max_vals - min_vals;
    end

    b = mean(vals, 'omitnan');
    err = std(vals, 'omitnan')/ sum(~isnan(vals)); 
    bars = [bars; b];
    errbars = [errbars; err];

end

figure
xs = [1,2,3,4];
%bar(xs, bars)
%hold on
errorbar(xs, bars, errbars, errbars)
legend(strjoin(legend_entry, ", "))

xlabel("task types")
xticks([1 2 3 4])
xticklabels(tasks)
ylabel(bar_type)
title(strcat(bar_type, " across tasks by trial"))
savefig(strcat(save_to,"across_tasks_by_trial_", strrep(bar_type, " ", "_"), ".fig"))
end