function bar_plots_by_subject(tasks,tables,bar_type,save_to,is_hr)

bars = [];
errbars = [];
legend_entry = [];
for i = 1:length(tasks)
    tit = tasks{i};
    table = tables{i};

    subjects = unique(table.subjectidnumber);
    legend_entry = [legend_entry; tit + ": n = " + string(length(subjects))];

    curr_bars = [];
    curr_errbars = [];

    for s = 1:length(subjects)
        subid = subjects(s);
        if is_hr
            curr_table = table(table.subjectidnumber == string(subid{1}) & str2double(table.number_of_timesteps(:)) > 9,:);
        else
            curr_table = table(table.subjectidnumber == string(subid{1}),:);
        end

        if ~isempty(curr_table)
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
            curr_bars = [curr_bars; b];
        end
    end

    bars = [bars; mean(curr_bars, 'omitnan')];
    errbars = [errbars; std(curr_bars, 'omitnan')/sum(~isnan(curr_bars))];
   
end

figure
xs = [1,2,3,4];
%bar(xs, bars)
%hold on

errorbar(xs, bars, errbars, errbars)

xlabel("task types")
xticks([1 2 3 4])
xticklabels(tasks)
ylabel(bar_type)
legend(strjoin(legend_entry, ", "))
title(strcat(bar_type, " across tasks by subject"))
savefig(strcat(save_to,"across_tasks_by_subject_", strrep(bar_type, " ", "_"), ".fig"))
end